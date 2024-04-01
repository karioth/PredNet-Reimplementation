import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

class SequenceGenerator(Sequence):
    def __init__(self, data_file, source_file, nt, sequence_start_mode='all'):
        
        self.start_mode = sequence_start_mode
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
        self.N_sequences = len(self.possible_starts)
    
    def __len__(self):
        return self.N_sequences
    
    def __getitem__(self, index):
        # Randomly select a starting index
        idx = self.possible_starts[index]
        # Generate a single sequence
        sequence = self.preprocess(self.X[idx:idx+self.nt])
        target = 0.0
        return sequence, target

    # Preprocess the data (normalize)
    def preprocess(self, X):
        return X.astype(np.float32) / 255
    
    # Create all sequences (optional utility function)
    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all

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
