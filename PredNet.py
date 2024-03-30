import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras import backend as K

class PredNet_Cell(layers.Layer):
  ''' wraps the ConvLSTM2DCell to incorporate the targets (A), predictions (A_hat) and error (E) computations involved in a single step of a PredNet layer'''
  def __init__(self, stack_size, R_stack_size, A_filt_size, Ahat_filt_size, R_filt_size,**kwargs):
        super(PredNet_Cell, self).__init__(**kwargs)

     #Extract the necessary hyperparameters for this PredNet layer
        self.stack_size = stack_size
        self.R_stack_size = R_stack_size
        self.A_filt_size = A_filt_size
        self.Ahat_filt_size = Ahat_filt_size
        self.R_filt_size = Ahat_filt_size

      #Start builing the modules on this layer:
      #For a -- Computes the target, does not apply if we are at the bottom layer
        self.conv_a = layers.Conv2D(filters = self.stack_size, kernel_size = self.A_filt_size, padding='same', activation = 'relu')
        self.pool_a = layers.MaxPooling2D()

      #for a_hat -- computes the prediction
        self.conv_a_hat = layers.Conv2D(filters = self.stack_size, kernel_size = self.Ahat_filt_size, padding='same', activation = 'relu')

      #for r -- computes the representation used to make a prediction
        self.convlstmcell = ConvLSTM2DCell(filters=self.R_stack_size, kernel_size=self.R_filt_size, padding='same', activation='relu', strides=(1, 1))
        self.upsample = layers.UpSampling2D(size=(2, 2))

      #for e -- computes the negative and possitive error.
        self.substract = layers.Subtract()
        self.relu = layers.ReLU()

  @property
  def state_size(self):
    '''returns the state sizes at the corresponding layer'''
    r_state_size = self.R_stack_size
    c_state_size = self.R_stack_size
    e_state_size = self.stack_size*2 # E state is doubled to account for positive and negative error

    return (r_state_size, c_state_size, e_state_size)

  @property
  def output_size(self):
    return None

  def top_down(self, states, top_r = None):
    '''Custom top down call. It implements the top-down update sequence for this layer.
      Takes as argument the states computed here on the previous time-step and the top-down feedback (None if we are at the top layer).
      Uses them to compute the updated r and c states'''

    #Disentangle the states
    prev_r = states[0]
    prev_c = states[1]
    prev_e = states[2]

    if top_r is not None: # we up-sample the top-down feedback to match the the pool in the bottom-up update
      upsamp_r = self.upsample(top_r)
      inputs = tf.concat([prev_e, prev_r, upsamp_r], axis=-1) # we use the upsampled top down feedback, the previous error and r representation as inputs to the convlstm cell.
    else:
      inputs = tf.concat([prev_e, prev_r], axis=-1) # use only the previous error and r if we are at the top layer.

    new_r, conv_lstm_states = self.convlstmcell(inputs, [prev_r, prev_c]) # we pass the r and c states expected by the ConvLSTM2DCell along with the input.
    new_c = conv_lstm_states[1]

    return new_r, new_c #return the new new_r state to send as feedback downwards and new new_c state for use on the next time step.

  def call(self, error_input, new_r, bottom=False):
        ''' Bottom-up call. It implements the bottom-up update to compute the target (a), prediction (a_hat) and prediction error (new_e).
        Takes as argument the error from the layer below (or the frame at the bottom layer) and the new_r representation computed in the top-down update.'''

        if bottom: # we take the frame as the target
          a = error_input
        else: # we conv_a + pool over the prediction error forwarded from the layer below to get out target.
          a = self.conv_a(error_input)
          a = self.pool_a(a)

        a_hat = self.conv_a_hat(new_r) # we use the new_r representation to compute our prediction (a_hat) of the target.

        if bottom: # We apply clipping to set the at maximum pixel value (1)
          a_hat = tf.minimum(1.0, a_hat)

        #compute the positive error
        pos_error = self.substract([a, a_hat])
        pos_error = self.relu(pos_error)

        #compute the negative error
        neg_error = self.substract([a_hat, a])
        neg_error = self.relu(neg_error)

        new_e = tf.concat([pos_error, neg_error], axis=-1) #Concatenate them along the feature dimension to get the error response.

        if bottom: # we output the frame prediction as well
          frame_prediction = a_hat
          return new_e, frame_prediction

        return new_e # propagate error response foward to be used by the layer above
   def get_config(self):
        config = {
            'stack_size': self.stack_size,
            'R_stack_size': self.R_stack_size,
            'A_filt_size': self.A_filt_size,
            'Ahat_filt_size': self.Ahat_filt_size,
            'R_filt_size': self.R_filt_size
        }
        base_config = super().get_config()  # Include standard layer attributes
        
        return dict(list(base_config.items()) + list(config.items())
    

class StackPredNet(layers.StackedRNNCells):
    ''' Base Class for stacking PredNet Cells.
    Takes as input a list of PredNet_Cells and stacks them to handle their correct hierarchical interaction on every step of the sequence'''
    def __init__(self, **kwargs):
        super(StackPredNet, self).__init__(**kwargs)

        self.nb_layers = len(self.cells)

    def build(self, input_shape):
      ''' RNN class requires that the build method of the cell must define.
      Takes as input the shape of a frame. From it we build each component of the cells by modifying it to match their expected inputs from based on their hierarchical position'''

      # get the height and width dimensions
      nb_row = input_shape[1]
      nb_col = input_shape[2]

      for layer_index, cell in enumerate(self.cells): # iterate through each PredNetCell
        ds_factor = 2 ** layer_index # Adapt Input Shape based on hierarchical position

        if layer_index > 0: #build conv_a for non bottom layer, get the right feature dimension by multiplying from the the stack_size from the layer below.
          cell.conv_a.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, 2 * self.cells[layer_index -1].stack_size))

        cell.conv_a_hat.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, cell.R_stack_size)) # build conv_a_hat

        if layer_index < self.nb_layers - 1: # if not at the top, build convlstmcell accounting for top-down feedback
          cell.convlstmcell.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, cell.stack_size * 2 + cell.R_stack_size + self.cells[layer_index+1].R_stack_size))

        else: #build convlstmcell without top down feedback.
          cell.convlstmcell.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, cell.stack_size * 2 + cell.R_stack_size))

        cell.built = True # set cell to built

      self.built = True #set the stack to built


    def get_initial_state(self, inputs):
      ''' Takes as input the entire sequence and returns the initial set of zero initilazied states require to begin iterating.'''

      input_shape = inputs.shape
      # get the height and width dimensions
      init_nb_row = input_shape[2]
      init_nb_col = input_shape[3]

      base_initial_state = K.zeros_like(inputs)

      non_channel_axis =  -2

      for _ in range(2):
          base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)

      base_initial_state = K.sum(base_initial_state, axis=1) #should have shape (samples, nb_channels)

      initial_states = [[], [], []]  # initialize empty lists for 'r', 'c', and 'e' states
      states_to_pass = ['r', 'c', 'e'] # to iterate over

      for i, cell in enumerate(self.cells):
          layer_index = i
          for state_type_index, state_type in enumerate(states_to_pass):
              nb_row = init_nb_row // (2 ** layer_index)
              nb_col = init_nb_col // (2 ** layer_index)

              if state_type in ['r', 'c']:
                  stack_size = self.cells[layer_index].R_stack_size
              else:  # state_type == 'e'
                  stack_size = 2 * self.cells[layer_index].stack_size

              output_size = stack_size * nb_row * nb_col
              reducer = tf.zeros((input_shape[-1], output_size))
              initial_state = K.dot(base_initial_state, reducer)
              output_shp = (-1, nb_row, nb_col, stack_size)
              initial_state = K.reshape(initial_state, output_shp)
              initial_states[state_type_index].append(initial_state)

      return initial_states

    @property
    def state_size(self):
      ''' returns a list of the state sizes of each layer '''
      r_state_sizes = []
      c_state_sizes = []
      e_state_sizes = []

      for c in self.cells:
          r, c, e = c.state_size
          r_state_sizes.append(r)
          c_state_sizes.append(c)
          e_state_sizes.append(e)

      return [r_state_sizes, c_state_sizes, e_state_sizes]

    def call(self, input, states, training = False):
        ''' Equivalent to the step function in the original implementation. Handles the dynamics across cell layers of the PredNet'''
        # We disentangle the states
        prev_r_states = states[0]
        prev_c_states = states[1]
        prev_e_states = states[2]

        current_input = input # set the current input to be the frame

        #initialize list for the new states to be computed
        new_r_states = []
        new_c_states = []
        new_e_states = []

        all_error = None # Variable for tranining.

        #top down pass using the custom top_down call of each cell. We iterate in reverse and calculate the new new_r and new_c states:
        for l, cell in reversed(list(enumerate(self.cells))):
          layer_states = [prev_r_states[l], prev_c_states[l], prev_e_states[l]]

          if l == self.nb_layers - 1:
            new_r, new_c = cell.top_down(layer_states, top_r=None) # pass None as feedback if we are at the top layer.
          else:
            new_r, new_c = cell.top_down(layer_states, top_r = new_r) # pass the new_r just calculated for the layer above as top down feedback.

          #insert states on the list rather than appending to not get a reversed list.
          new_r_states.insert(0, new_r)
          new_c_states.insert(0, new_c)

        #bottom_up pass, we iterate normarly calling every cell with the current input and the just calculated new_r representation corresponding to that layer.
        for l, cell in enumerate(self.cells):
          new_r = new_r_states[l]
          if l == 0:  # Bottom layer
            error, frame_prediction = cell(current_input, new_r, bottom=True)
          else:
            error = cell(current_input, new_r)

          current_input = error #pass the error just computed forward as the input to the next layer.

          new_e_states.append(error)
          layer_error = tf.reduce_mean(tf.keras.layers.Flatten()(error), axis=-1, keepdims=True)
          all_error = layer_error if l == 0 else tf.concat((all_error, layer_error), axis=-1) # add the layer_error to the all_error output for traning.

        new_states_per_layer = [new_r_states, new_c_states, new_e_states] # Make a list of the new states for the next time step

        if training:
           output = all_error # during traning we output all error to train on reducing error to 0
        else:
           output = frame_prediction # for inference we output the frame prediction.

        return output, new_states_per_layer

class PredNet(RNN):
  """Base class for PredNet Architectures. Modifies the ConvRNN2D class to allow (PredNet) cell stacking.
  Only minor changes to the build function and offloading the get_initial_states to the StackPredNet class.
  Arguments:
    cell: A list of PredNet-like cells. A RNN cell is a class that has:
      - a `call(input_at_t, states_at_t)` method, returning
        `(output_at_t, states_at_t_plus_1)`. The call method of the
        cell can also take the optional argument `constants`, see
        section "Note on passing external constants" below.
      - a `state_size` attribute. This can be a single integer
        (single state) in which case it is
        the number of channels of the recurrent state
        (which should be the same as the number of channels of the cell
        output). This can also be a list/tuple of integers
        (one size per state). In this case, the first entry
        (`state_size[0]`) should be the same as
        the size of the cell output.
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    input_shape: Use this argument to specify the shape of the
      input when this layer is the first one in a model.

  Call arguments:
    inputs: A 5D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is for use with cells that use dropout.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
    constants: List of constant tensors to be passed to the cell at each
      timestep.

  Input shape:
    5D tensor with shape:
    `(samples, timesteps, rows, cols, channels)`.

  Output shape:
    - If `return_state`: a list of tensors. The first tensor is
      the output. The remaining tensors are the last states,
      each 4D tensor with shape:
      `(samples, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)`
      if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    - If `return_sequences`: 5D tensor with shape:
      `(samples, timesteps, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 5D tensor with shape:
      `(samples, timesteps, new_rows, new_cols, filters)`
      if data_format='channels_last'.
    - Else, 4D tensor with shape:
      `(samples, filters, new_rows, new_cols)`
      if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)`
      if data_format='channels_last'.

  Masking:
    This layer supports masking for input data with a variable number
    of timesteps.

  Note on using statefulness in RNNs:
    You can set RNN layers to be 'stateful', which means that the states
    computed for the samples in one batch will be reused as initial states
    for the samples in the next batch. This assumes a one-to-one mapping
    between samples in different successive batches.
    To enable statefulness:
      - Specify `stateful=True` in the layer constructor.
      - Specify a fixed batch size for your model, by passing
         - If sequential model:
            `batch_input_shape=(...)` to the first layer in your model.
         - If functional model with 1 or more Input layers:
            `batch_shape=(...)` to all the first layers in your model.
            This is the expected shape of your inputs
            *including the batch size*.
            It should be a tuple of integers,
            e.g. `(32, 10, 100, 100, 32)`.
            Note that the number of rows and columns should be specified
            too.
      - Specify `shuffle=False` when calling fit().
    To reset the states of your model, call `.reset_states()` on either
    a specific layer, or on your entire model.

  Note on specifying the initial state of RNNs:
    You can specify the initial state of RNN layers symbolically by
    calling them with the keyword argument `initial_state`. The value of
    `initial_state` should be a tensor or list of tensors representing
    the initial state of the RNN layer.
    You can specify the initial state of RNN layers numerically by
    calling `reset_states` with the keyword argument `states`. The value of
    `states` should be a numpy array or list of numpy arrays representing
    the initial state of the RNN layer.

  Note on passing external constants to RNNs:
    You can pass "external" constants to the cell using the `constants`
    keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
    requires that the `cell.call` method accepts the same keyword argument
    `constants`. Such constants can be used to condition the cell
    transformation on additional static inputs (not changing over time),
    a.k.a. an attention mechanism.
  """

  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               output_mode = 'error',
               **kwargs):

    if isinstance(cell, (list, tuple)): # stack PredNet_Cells using StackPredNet
      cell = StackPredNet(cells = cell)

    super(PredNet, self).__init__(cell,
                                    return_sequences,
                                    return_state,
                                    go_backwards,
                                    stateful,
                                    unroll,
                                    **kwargs)

    self.input_spec = [InputSpec(ndim=5)]
    self.states = None
    self._num_constants = None
    self.output_mode = output_mode

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    ''' Assumes traning (error) mode, based on the authors original code'''
    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    cell = self.cell
    if self.output_mode == 'prediction':
        out_shape = input_shape[2:]
    elif self.output_mode == 'error':
        out_shape = (cell.nb_layers,)
    elif self.output_mode == 'all':
        out_shape = (np.prod(input_shape[2:]) + cell.nb_layers,)

    if self.return_sequences:
        output_shape = (input_shape[0], input_shape[1]) + out_shape
    else:
        output_shape = (input_shape[0],) + out_shape
    return output_shape

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]  # pylint: disable=E1130
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

    # Change to allow the cell to build before we set or validate state_spec

    step_input_shape = (input_shape[0],) + input_shape[2:]
    if constants_shape is not None:
      self.cell.build([step_input_shape] + constants_shape)
    else:
      self.cell.build(step_input_shape)

    # set or validate state_spec
    if hasattr(self.cell.state_size, '__len__'):
      state_size = self.cell.state_size # [[r's],[c's],[e's]]
    else:
      state_size = self.cell.state_size

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      ch_dim = 3 #assume channel's last
      
      if [spec.shape[ch_dim] for spec in self.state_spec] != state_size:
        raise ValueError(
            'An initial_state was passed that is not compatible with '
            '`cell.state_size`. Received `state_spec`={}; '
            'However `cell.state_size` is '
            '{}'.format([spec.shape for spec in self.state_spec],
                        self.cell.state_size))
    else:
       self.state_spec = [] # Change to allow for nested state_spect due to stacking
       for state in state_size:
          self.state_spec.append([InputSpec(shape=(None, None, None, dim))
                            for dim in state])
    if self.stateful:
      self.reset_states()
    self.built = True

  def get_initial_state(self, inputs):
    '''Changed to simply use the get_initial_state function from the StackPredNet class'''
    initial_state = self.cell.get_initial_state(inputs)
    return initial_state

  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    inputs, initial_state, constants = self._process_inputs(
        inputs, initial_state, constants)

    if isinstance(mask, list):
      mask = mask[0]
    timesteps = K.int_shape(inputs)[1]

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
        states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type
        return self.cell.call(inputs, states, constants=constants, **kwargs)
    else:
      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

    last_output, outputs, states = K.rnn(step,
                                         inputs,
                                         initial_state,
                                         constants=constants,
                                         go_backwards=self.go_backwards,
                                         mask=mask,
                                         input_length=timesteps)
    if self.stateful:
      updates = [
          K.update(self_state, state)
          for self_state, state in zip(self.states, states)
      ]
      self.add_update(updates)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return [output] + states
    else:
      return output

  def reset_states(self, states=None):
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    input_shape = self.input_spec[0].shape
    state_shape = self.compute_output_shape(input_shape)
    if self.return_state:
      state_shape = state_shape[0]
    if self.return_sequences:
      state_shape = state_shape[:1].concatenate(state_shape[2:])
    if None in state_shape:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a '
                       '`batch_shape` argument to your Input layer.\n'
                       'The same thing goes for the number of rows and '
                       'columns.')

    # helper function
    def get_tuple_shape(nb_channels):
      result = list(state_shape)
      if self.cell.data_format == 'channels_first':
        result[1] = nb_channels
      elif self.cell.data_format == 'channels_last':
        result[3] = nb_channels
      else:
        raise KeyError
      return tuple(result)

    # initialize state if None
    if self.states[0] is None:
      if hasattr(self.cell.state_size, '__len__'):
        self.states = [K.zeros(get_tuple_shape(dim))
                       for dim in self.cell.state_size]
      else:
        self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
    elif states is None:
      if hasattr(self.cell.state_size, '__len__'):
        for state, dim in zip(self.states, self.cell.state_size):
          K.set_value(state, np.zeros(get_tuple_shape(dim)))
      else:
        K.set_value(self.states[0],
                    np.zeros(get_tuple_shape(self.cell.state_size)))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, ' +
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if hasattr(self.cell.state_size, '__len__'):
          dim = self.cell.state_size[index]
        else:
          dim = self.cell.state_size
        if value.shape != get_tuple_shape(dim):
          raise ValueError('State ' + str(index) +
                           ' is incompatible with layer ' +
                           self.name + ': expected shape=' +
                           str(get_tuple_shape(dim)) +
                           ', found shape=' + str(value.shape))
        # TODO(anjalisridhar): consider batch calls to `set_value`.
        K.set_value(state, value)
