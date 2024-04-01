class SequenceGenerator:
    def __init__(self, data_file, source_file, nt, sequence_start_mode='all'):
        # Open the data file and load the data
        with h5py.File(data_file, 'r') as f:
            key = list(f.keys())[0]
            self.X = f[key][:]
        
        # Open the source file and load the sources (e.g., video indices or identifiers)
        with h5py.File(source_file, 'r') as f:
            key = list(f.keys())[0]
            self.sources = f[key][:]
        
        self.nt = nt  # Number of timesteps per sequence
        self.im_shape = self.X[0].shape  # Image shape
        
        # Determine possible start indices for sequences
        if sequence_start_mode == 'all':
            self.possible_starts = np.array([i for i in range(len(self.sources) - self.nt + 1) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif sequence_start_mode == 'unique':
            self.possible_starts = self._calculate_unique_starts()

        
        # Create a TensorFlow Dataset using from_generator
        self.dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(self.nt,) + self.im_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            ))
    
    def _generator(self):
        while True:
            # Randomly select a starting index
            start = np.random.choice(self.possible_starts)
            
            # Generate a single sequence
            sequence = self.preprocess(self.X[start:start+self.nt])
            target = 0.0
            
            yield sequence, target
    
    # Preprocess the data (normalize)
    def preprocess(self, X):
        return X.astype(np.float32) / 255
    
    def get_dataset(self):
        return self.dataset

    def _calculate_unique_starts(self):
        curr_location = 0
        possible_starts = []
        while curr_location < len(self.sources) - self.nt + 1:
            if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                possible_starts.append(curr_location)
                curr_location += self.nt
            else:
                curr_location += 1
        return np.array(possible_starts)
