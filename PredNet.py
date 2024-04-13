import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras import backend as K

@tf.keras.saving.register_keras_serializable(package="PredNet_Cell")
class PredNet_Cell(layers.Layer):
    """
    This class represents a cell of the Predictive Coding Network (PredNet) which is responsible for a single timestep.
    It encapsulates the computations of targets (A), predictions (A_hat), and errors (E) within one PredNet layer.

    Attributes:
        stack_size (int): Number of channels in the target/prediction layers.
        R_stack_size (int): Number of channels in the representation layer.
        A_filt_size (tuple): Kernel size for the convolutional layer computing the targets.
        Ahat_filt_size (tuple): Kernel size for the convolutional layer computing the predictions.
        R_filt_size (tuple): Kernel size for the ConvLSTM2D layer computing the representation.
    """

    def __init__(self, stack_size, R_stack_size, A_filt_size, Ahat_filt_size, R_filt_size, **kwargs):
        super(PredNet_Cell, self).__init__(**kwargs)
        self.stack_size = stack_size
        self.R_stack_size = R_stack_size
        self.A_filt_size = A_filt_size
        self.Ahat_filt_size = Ahat_filt_size
        self.R_filt_size = R_filt_size

        # Target computation: Convolutional layer (activation: ReLU), does not build if we are at the bottom layer
        self.conv_a = layers.Conv2D(filters=self.stack_size, kernel_size=self.A_filt_size, padding='same', activation='relu')
        self.pool_a = layers.MaxPooling2D()

        # Prediction computation: Convolutional layer (activation: ReLU)
        self.conv_a_hat = layers.Conv2D(filters=self.stack_size, kernel_size=self.Ahat_filt_size, padding='same', activation='relu')

        # Representation computation: ConvLSTM2DCell (activation: tanh, recurrent_activation: hard_sigmoid)
        self.convlstmcell = ConvLSTM2DCell(filters=self.R_stack_size, kernel_size=self.R_filt_size, padding='same', activation='tanh', recurrent_activation='hard_sigmoid')
        self.upsample = layers.UpSampling2D(size=(2, 2))

        # Error computation: Subtract and ReLU for positive and negative error components
        self.subtract = layers.Subtract()
        self.relu = layers.ReLU()


    @property
    def state_size(self):
        """ Returns the sizes of the various states maintained by the layer: representation, cell, and error states. """
        r_state_size = self.R_stack_size  # Representation state size
        c_state_size = self.R_stack_size  # Cell state size (for LSTM)
        e_state_size = self.stack_size * 2  # Error state size (positive and negative)
        return (r_state_size, c_state_size, e_state_size)

    @property
    def output_size(self):
        """ The layer has no fixed output size due to its dynamic nature. """
        return None

    def top_down(self, states, top_r=None):
      """
      Performs the top-down update sequence for this layer using previous states and top-down feedback.
      
      Args:
          states (tuple): The previous states from this layer.
          top_r (Tensor, optional): The top-down feedback from the layer above, None if this is the top layer.
  
      Returns:
          tuple: Updated states (new_r, new_c) after processing top-down feedback and previous states.
      """
      # Unpack the previous states
      prev_r = states[0]
      prev_c = states[1]
      prev_e = states[2]
  
      # If there's top-down feedback, upsample and concatenate it with previous states
      if top_r is not None:
          upsamp_r = self.upsample(top_r)
          inputs = tf.concat([prev_e, prev_r, upsamp_r], axis=-1) # we use the upsampled top down feedback, the previous error and r representation as inputs to the convlstm cell.
      else:
          inputs = tf.concat([prev_e, prev_r], axis=-1) # use only the previous error and r if we are at the top layer.
  
      # Update representation and cell states using the ConvLSTM2DCell
      new_r, conv_lstm_states = self.convlstmcell(inputs, [prev_r, prev_c]) # we pass the r and c states expected by the ConvLSTM2DCell along with the input.
      new_c = conv_lstm_states[1]
  
      return new_r, new_c #return the new new_r state to send as feedback downwards and new new_c state for use on the next time step.

    def call(self, error_input, new_r, bottom=False):
        """
        Performs the bottom-up update to compute targets (a), predictions (a_hat), and prediction errors (new_e).

        Args:
            error_input (Tensor): The error input from the layer below, or the input frame at the bottom layer.
            new_r (Tensor): The representation computed in the top-down update.
            bottom (bool): Flag indicating whether this is the bottom layer.

        Returns:
            Tensor: The new error computed, and optionally the frame prediction if this is the bottom layer.
        """
        # Compute the target using the error input
        if bottom:
            a = error_input  # Directly use the input frame as the target at the bottom layer
        else:
            a = self.conv_a(error_input) # Apply convolutions and pooling to the error input
            a = self.pool_a(a)  

        # Compute the prediction using the new representation state
        a_hat = self.conv_a_hat(new_r)
        if bottom:
            a_hat = tf.minimum(1.0, a_hat)  # Clip predictions to a maximum of 1

        # Compute positive and negative prediction errors
        pos_error = self.subtract([a, a_hat])
        pos_error = self.relu(pos_error)

        neg_error = self.subtract([a_hat, a])
        neg_error = self.relu(neg_error)
        new_e = tf.concat([pos_error, neg_error], axis=-1) #Concatenate them along the feature dimension to get the error response.

        # Optionally return the frame prediction at the bottom layer
        if bottom:
            return new_e, a_hat

        return new_e # propagate error response foward to be used by the layer above

    def get_config(self):
        """
        Returns the configuration of the PredNet_Cell for Keras model serialization.
        """
        config = {
          'stack_size': self.stack_size,
          'R_stack_size': self.R_stack_size,
          'A_filt_size': self.A_filt_size,
          'Ahat_filt_size': self.Ahat_filt_size,
          'R_filt_size': self.R_filt_size}
        
        base_config = super().get_config() # Include standard layer attributes
    
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of PredNet_Cell from a configuration dictionary.
        """
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="StackedRNNCells")
class StackPredNet(layers.StackedRNNCells):
    """
    A class that handles the stacking of PredNet Cells. It orchestrates the hierarchical interactions
    between different layers in a PredNet architecture during sequence processing.

    Extends:
        layers.StackedRNNCells: A class that can stack multiple RNN cells.

    Initialization:
        cells (list of PredNet_Cell): The PredNet cells to be stacked.
    """

    def __init__(self, **kwargs):
        super(StackPredNet, self).__init__(**kwargs)
        self.nb_layers = len(self.cells)  # Store the number of layers

    def build(self, input_shape):
        """
        Build each component of the cells based on their position in the hierarchy.

        Args:
            input_shape (tuple): The shape of the input frame.

        Explanation:
            This method adjusts the input shape for each layer based on its depth in the stack,
            taking into account the downsampling factor and the specific needs for convolutional and LSTM components.
        """
        # Extract spatial dimensions from input shape
        nb_row = input_shape[1]
        nb_col = input_shape[2] 

        for layer_index, cell in enumerate(self.cells):
            ds_factor = 2 ** layer_index  # Calculate downsampling factor for this layer

            if layer_index > 0:  # Adjust the input shape and build the conv_a for non-bottom layers
                cell.conv_a.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, 2 * self.cells[layer_index - 1].stack_size))

            # Build the conv_a_hat using the representation size
            cell.conv_a_hat.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, cell.R_stack_size))

            # Adjust the input shape and build the ConvLSTM2DCell for each layer considering top-down feedback
            if layer_index < self.nb_layers - 1:
                cell.convlstmcell.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, cell.stack_size * 2 + cell.R_stack_size + self.cells[layer_index + 1].R_stack_size))
            else:  # For the top layer, there is no top-down feedback
                cell.convlstmcell.build((input_shape[0], nb_row // ds_factor, nb_col // ds_factor, cell.stack_size * 2 + cell.R_stack_size))

            cell.built = True  # Mark the cell as built

        self.built = True  # Mark the stack as built

    def get_initial_state(self, inputs):
        """
        Generates the initial set of zero-initialized states for the sequence processing.

        Args:
            inputs (Tensor): The input tensor containing the entire sequence.

        Returns:
            list: The initial states for each cell in the stack.
        """
        input_shape = inputs.shape
        # Get height and width dimensions
        init_nb_row = input_shape[2]
        init_nb_col = input_shape[3]

        base_initial_state = K.zeros_like(inputs)  # Start with a base state of zeros

        non_channel_axis =  -2

        # Reduce across spatial dimensions to get a base state with the shape (samples, nb_channels)
        for _ in range(2):
            base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)

        base_initial_state = K.sum(base_initial_state, axis=1)  # Sum over time dimension

        initial_states = [[], [], []]  # Initialize lists for 'r', 'c', 'e' states
        states_to_pass = ['r', 'c', 'e']  # Define the types of states

        for i, cell in enumerate(self.cells):
            layer_index = i
            for state_type_index, state_type in enumerate(states_to_pass):
                nb_row = init_nb_row // (2 ** layer_index)
                nb_col = init_nb_col // (2 ** layer_index)

                # Determine the size of the state based on the cell's configuration
                if state_type in ['r', 'c']:
                    stack_size = self.cells[layer_index].R_stack_size
                else:  # 'e' state
                    stack_size = 2 * self.cells[layer_index].stack_size

                # Prepare initial state tensor for this state type
                output_size = stack_size * nb_row * nb_col
                reducer = tf.zeros((input_shape[-1], output_size))
                initial_state = K.dot(base_initial_state, reducer)
                output_shp = (-1, nb_row, nb_col, stack_size)
                initial_state = K.reshape(initial_state, output_shp)
                initial_states[state_type_index].append(initial_state)

        return initial_states


    @property
    def state_size(self):
        """
        Returns the state sizes for each of the layers in the stack, organized by state type (r, c, e).
        This helps in preparing and initializing the states when the network is executed.
    
        Returns:
            list: A list containing three lists, one for each state type (r, c, e), each containing the state sizes for the corresponding layer.
        """
        r_state_sizes, c_state_sizes, e_state_sizes = [], [], []
    
        for cell in self.cells:
            r, c, e = cell.state_size
            r_state_sizes.append(r)
            c_state_sizes.append(c)
            e_state_sizes.append(e)
    
        return [r_state_sizes, c_state_sizes, e_state_sizes]
    
    def call(self, inputs, states, training=False):
        """
        Process input through the stacked PredNet cells, handling both top-down and bottom-up dynamics.
    
        Args:
            inputs (Tensor): The input tensor at the current timestep.
            states (list): The states from the previous timestep.
            training (bool): Flag to determine whether the network is in training mode or inference mode.
    
        Returns:
            tuple: Contains the output (either the prediction error during training or the predicted frame during inference) and the new states.
        """
        # Disentangle the states into separate lists for each state type
        prev_r_states = states[0]
        prev_c_states = states[1]
        prev_e_states = states[2]
    
        current_input = inputs # Set the current input to be the frame
        # Initialize list for the new states to be computed
        new_r_states, new_c_states, new_e_states = [], [], []
    
        all_error = None  # Variable for training error accumulation
    
        # Top-down pass: Update states from top to bottom. We iterate in reverse and calculate the new new_r and new_c states
        for l, cell in reversed(list(enumerate(self.cells))):
            layer_states = [prev_r_states[l], prev_c_states[l], prev_e_states[l]]

            if l == self.nb_layers - 1:
              new_r, new_c = cell.top_down(layer_states, top_r=None) # pass None as feedback if we are at the top layer.
            else:
              new_r, new_c = cell.top_down(layer_states, top_r = new_r) # pass the new_r just calculated for the layer above as top down feedback.

            #insert states on the list rather than appending to not get a reversed list.
            new_r_states.insert(0, new_r)
            new_c_states.insert(0, new_c)
    
        # Bottom-up pass: Process errors and update states.
        for l, cell in enumerate(self.cells):
            new_r = new_r_states[l]
            if l == 0:  # Bottom layer processes input frame directly
                error, frame_prediction = cell(current_input, new_r, bottom=True)
            else:
                error = cell(current_input, new_r)
    
            current_input = error  # Pass the error up to the next layer
            new_e_states.append(error)
    
            if training:  # Accumulate errors for training
                layer_error = tf.reduce_mean(tf.keras.layers.Flatten()(error), axis=-1, keepdims=True)
                all_error = layer_error if l == 0 else tf.concat((all_error, layer_error), axis=-1)
    
        # Collect new states for each layer for next time step
        new_states_per_layer = [new_r_states, new_c_states, new_e_states]
    
        # Decide output based on the mode (training vs. inference)
        output = all_error if training else frame_prediction
    
        return output, new_states_per_layer

      
@tf.keras.saving.register_keras_serializable(package="PredNet")
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
    """
    Initializes the PredNet model with a stack of PredNet cells.

    Args:
        cell (list of PredNet_Cell): A list of PredNet-like cells or a single cell.
        return_sequences (bool): Whether to return the full sequence of outputs or just the last output.
        return_state (bool): Whether to return the last state in addition to the output.
        go_backwards (bool): Whether to process the input sequence backwards.
        stateful (bool): Whether the states in the network should be maintained across batches.
        unroll (bool): Whether the network should unroll the RNN loops.
        output_mode (str): Output mode of the network, could be 'error', 'prediction', or 'all'.
        **kwargs: Additional keyword arguments for the RNN layer.

    Note:
        If `cell` is a list of PredNet cells, it is converted into a `StackPredNet` for hierarchical processing.
    """

    if isinstance(cell, (list, tuple)): # If cells are provided as a list, stack them using StackPredNet
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
    """
    Computes the output shape of the network based on the input shape and the mode of operation.

    Args:
        input_shape (tuple): Shape of the input batch (including the batch size).

    Returns:
        tuple: The shape of the output which varies based on `output_mode` and other configurations.

    Note:
        Assumes training (error) mode, based on the authors' original code. It modifies the output shape based on the selected `output_mode`.
    """
    if isinstance(input_shape, list):
      input_shape = input_shape[0] # Handle nested input shapes

    cell = self.cell
    if self.output_mode == 'prediction':
        out_shape = input_shape[2:]
    elif self.output_mode == 'error':
        out_shape = (cell.nb_layers,) # Output one error per layer
    elif self.output_mode == 'all':
        out_shape = (np.prod(input_shape[2:]) + cell.nb_layers,)

    if self.return_sequences:
        output_shape = (input_shape[0], input_shape[1]) + out_shape
    else:
        output_shape = (input_shape[0],) + out_shape
    return output_shape

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    """
    Builds the PredNet architecture by initializing layers based on the input shape.

    Args:
        input_shape (tuple or list): Shape of the input or list of shapes if initial states and constants are provided.

    Note:
        Adjusts the `input_spec` to account for the batch size if `stateful` is True and ensures that all cell states are properly initialized.
    """
    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.

    # Handle the presence of constants in the input
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
    """
    Utilizes the StackPredNet's method to obtain initial states for the network.

    Args:
        inputs (Tensor): The initial inputs to the network, used to determine the shape of the initial states.

    Returns:
        list of Tensors: The initial states for the network's cells.
    """
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
