import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

class SequenceGenerator(Sequence):
    # Initialization of the generator
    def __init__(self, data_file, source_file, nt, batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None):
        # Open the data file and load the data
        with h5py.File(data_file, 'r') as f:
            key = list(f.keys())[0]
            self.X = f[key][:]
        # Open the source file and load the sources (e.g., video indices or identifiers)
        with h5py.File(source_file, 'r') as f:
            key = list(f.keys())[0]
            self.sources = f[key][:]

        # Set various parameters for data generation
        self.nt = nt  # Number of timesteps per sequence
        self.batch_size = batch_size
        # Ensure valid sequence_start_mode and output_mode values
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.sequence_start_mode = sequence_start_mode
        self.output_mode = output_mode

        self.im_shape = self.X[0].shape  # Image shape

        # Determine possible start indices for sequences
        if self.sequence_start_mode == 'all':
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':
            self.possible_starts = self._calculate_unique_starts()

        # Optionally shuffle the sequence starts
        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        # Limit the number of sequences if specified
        if N_seq is not None and len(self.possible_starts) > N_seq:
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)  # Total number of sequences

    # Define __len__ to make this class iterable
    def __len__(self):
        return self.N_sequences // self.batch_size

   # Define __getitem__ to make this class subscriptable
    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        idx_list = self.possible_starts[start:end]

        batch_x = np.zeros((self.batch_size, self.nt) + self.im_shape, np.float32)  # Allocate batch_x
        for i, idx in enumerate(idx_list):
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])  # Create sequences
        # Set batch_y depending on the output_mode
        if self.output_mode == 'error':
            batch_y = np.zeros(self.batch_size, np.float32)
        elif self.output_mode == 'prediction':
            batch_y = batch_x

        return batch_x, batch_y

    # Preprocess the data (normalize)
    def preprocess(self, X):
        return X.astype(np.float32) / 255

    # Create all sequences (optional utility function)
    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all

    # Calculate unique sequence starts (helper method for unique mode)
    def _calculate_unique_starts(self):
        curr_location = 0
        possible_starts = []
        while curr_location < self.X.shape[0] - self.nt + 1:
            if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                possible_starts.append(curr_location)
                curr_location += self.nt
            else:
                curr_location += 1
        return possible_starts
