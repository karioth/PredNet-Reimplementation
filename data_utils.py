import tensorflow as tf
import h5py
import numpy as np
import imageio
import IPython.display as display
import tensorflow as tf
import matplotlib.pyplot as plt

class SequenceGenerator: #used for hkl data files like used in the original data pipeline. 
    def __init__(self, data_file, source_file, nt, sequence_start_mode='all', shuffle=False):
        
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
        
        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        
        self.dataset = tf.data.Dataset.from_generator(self.__generator,
                                                      output_signature=(tf.TensorSpec(shape=(self.nt,) + self.im_shape, dtype=tf.float32)
                                                                        ))
    def __generator(self):
        while True:
            for idx in self.possible_starts:
                sequence = self.X[idx:idx+self.nt]
                yield sequence
    
    # Create all sequences (optional utility function for testing predictions)
    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = (self.X[idx:idx+self.nt]) / 255.0 # normalize 
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
        
    def get_dataset(self):
        return self.dataset



def visualize_sequences_as_gif(dataset, num=3):
    """
    Visualize a sequences of images (a video snippet) as an animated GIF with looping.
    Args: 
    dataset: dataset to get the sequences from, should have shape (batch_size, nt, height, width, n_channels)
    num: number of sequences to be plotted.  
    """
    #Fetch sequence from the dataset
    for sequence, target in dataset.take(num):
        first_sequence = (sequence[0][0].numpy() * 255).astype(np.uint8)
        # Create an animated GIF with looping
        with imageio.get_writer('sequence.gif', mode='I', duration=0.3, loop=0) as writer:
            for image in sequence:
                writer.append_data(image)
        # Load the GIF and display it
        with open('sequence.gif', 'rb') as f:
            display.display(display.Image(data=f.read(), format='png'))
        
        print("Shape of the sequence:", first_sequence.shape)
        fig, axes = plt.subplots(1, sequence_length, figsize=(20, 2))
        for i, ax in enumerate(axes):
            ax.imshow(first_sequence[i].astype("uint8"))
            ax.axis('off')
        plt.show()


def visualize_sequence(dataset, how_many=3, sequence_length=10):
    """
    Visualizes a sequence of images from a dataset.

    Args:
        dataset (tf.data.Dataset): The dataset containing the image sequences.
        how_many (int, optional): The number of sequences to visualize. Defaults to 3
        sequence_length (int, optional): The length of each sequence. Defaults to 10.

    Returns:
        None
    """
    # Fetch sequence from the dataset
    for sequence, target in dataset.take(how_many):
        # Assuming the dataset yields batches of shape [batch_size, sequence_length, height, width, channels]
        first_sequence = (sequence[0][0].numpy() * 255).astype(np.uint8)  # Convert the first sequence to a NumPy array for visualization
        # to visualize the first sequence in the batch
        visualize_sequence_as_gif(first_sequence)
        # `first_sequence` shape is (sequence_length, image_height, image_width, channels)
        print("Shape of the sequence:", first_sequence.shape)

        fig, axes = plt.subplots(1, sequence_length, figsize=(20, 2))
        for i, ax in enumerate(axes):
            ax.imshow(first_sequence[i].astype("uint8"))
            ax.axis('off')
        plt.show()
        
