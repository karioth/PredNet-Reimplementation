import tensorflow as tf
import os
import h5py
import numpy as np
import imageio
import IPython.display as display
import matplotlib
matplotlib.use('Agg')  # Configure matplotlib to use 'Agg' backend for compatibility reasons.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class SequenceGenerator:
    """
    A class to generate sequences from hierarchical data format (HDF5) files used for training models like PredNet.

    Attributes:
        data_file (str): The file path for the data in HDF5 format.
        source_file (str): The file path for the source identifiers in HDF5 format.
        nt (int): Number of time steps per sequence.
        sequence_start_mode (str): Strategy to select the start index of sequences ('all' or 'unique').
        shuffle (bool): If True, shuffle the sequence start indices.

    Methods:
        __generator: Private generator method to yield sequences for TensorFlow dataset.
        create_all: Create and return all possible sequences normalized by 255.
        _calculate_unique_starts: Calculate unique start indices for sequences to ensure no overlap.
        get_dataset: Get the TensorFlow dataset of sequences.
    """

    def __init__(self, data_file, source_file, nt, sequence_start_mode='all', shuffle=False):
        """
        Initialize the SequenceGenerator class with data, source files, and configuration.

        Parameters:
            data_file (str): Path to the HDF5 file containing the data.
            source_file (str): Path to the HDF5 file containing the source identifiers.
            nt (int): Number of timesteps to include in each sequence.
            sequence_start_mode (str): Mode to determine the starting points of sequences. Defaults to 'all'.
            shuffle (bool): Whether to shuffle the starting points of sequences. Defaults to False.
        """
        self.nt = nt
        self.start_mode = sequence_start_mode
        
        # Load data and sources from HDF5 files
        with h5py.File(data_file, 'r') as f:
            key = list(f.keys())[0]
            self.X = f[key][:]
        with h5py.File(source_file, 'r') as f:
            key = list(f.keys())[0]
            self.sources = f[key][:]
        
        self.im_shape = self.X[0].shape  # Determine the shape of each image frame

        # Determine sequence start indices based on the selected mode
        if sequence_start_mode == 'all':
            self.possible_starts = np.array([i for i in range(len(self.sources) - nt + 1) if self.sources[i] == self.sources[i + nt - 1]])
        elif sequence_start_mode == 'unique':
            self.possible_starts = self._calculate_unique_starts()
        
        self.N_sequences = len(self.possible_starts)  # Total number of possible sequences

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        
        self.dataset = tf.data.Dataset.from_generator(self.__generator,
                                                      output_signature=(tf.TensorSpec(shape=(nt,) + self.im_shape, dtype=tf.float32)))

    def __generator(self):
        """Generator function for creating sequences."""
        while True:
            for idx in self.possible_starts:
                sequence = self.X[idx:idx+self.nt]
                yield sequence

    def create_all(self):
        """
        Generate and return all sequences, normalized by dividing by 255.

        Returns:
            np.array: An array of normalized sequences.
        """
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.X[idx:idx+self.nt] / 255.0  # Normalize pixel values
        return X_all

    def _calculate_unique_starts(self):
        """
        Compute unique starting indices for the sequences to ensure that there is no overlap between them.

        Returns:
            np.array: Array of unique start indices.
        """
        curr_location = 0
        possible_starts = []
        while curr_location < len(self.sources) - self.nt + 1:
            if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                possible_starts.append(curr_location)
                curr_location += self.nt  # Move by 'nt' steps to ensure no overlap
            else:
                curr_location += 1
        return np.array(possible_starts)
    
    def get_dataset(self):
        """
        Get the TensorFlow dataset containing sequences.

        Returns:
            tf.data.Dataset: A TensorFlow dataset object.
        """
        return self.dataset




def visualize_sequence_as_gif(sequence):
    """
    Visualize a sequence of images (typically frames of a video) as an animated GIF with looping.

    Args:
        sequence (array-like): A sequence of images to be visualized as an animated GIF.

    Notes:
        The generated GIF is temporarily stored, displayed, and then deleted.
    """
    # Create an animated GIF with looping
    with imageio.get_writer('sequence.gif', mode='I', duration=0.3, loop=0) as writer:
        for image in sequence:
            writer.append_data(image)

    # Load the GIF and display it using IPython display tools
    with open('sequence.gif', 'rb') as f:
        display.display(display.Image(data=f.read(), format='png'))
    # Delete the gif file after displaying it to free up space
    os.remove('sequence.gif')


def visualize_sequence(dataset, how_many=3, sequence_length=10):
    """
    Visualize several sequences from a TensorFlow dataset as images and animated GIFs.

    Args:
        dataset (tf.data.Dataset): The dataset from which sequences will be visualized.
        how_many (int, optional): Number of sequences to visualize. Defaults to 3.
        sequence_length (int, optional): Length of each sequence to visualize. Defaults to 10.

    Returns:
        None
    """
    # Fetch and visualize specified number of sequences from the dataset
    for sequence in dataset.take(how_many):
        # Extract the first sequence in the batch for visualization
        first_sequence = (sequence[0][0].numpy() * 255).astype(np.uint8)  # Convert to uint8 for visualization
        visualize_sequence_as_gif(first_sequence)  # Display as GIF

        # Display each frame as a subplot
        fig, axes = plt.subplots(1, sequence_length, figsize=(20, 2))
        for i, ax in enumerate(axes):
            ax.imshow(first_sequence[i])
            ax.axis('off')
        plt.show()
        plt.close()

def evaluate_mse(X_test, X_hat, X_hat_ori=None):
    """
    Evaluate the mean squared error (MSE) for model predictions against the test data.

    Args:
        X_test (np.array): The ground truth test data.
        X_hat (np.array): The predicted data from the model.
        X_hat_ori (np.array, optional): Predictions from the original model for comparison.

    Returns:
        tuple: Returns a tuple containing MSE of the previous frame, current model, and optionally the original model.
    """
    # Calculate MSE between test data and predictions, excluding the first frame
    mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)
    mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)

    print("Previous Frame MSE: %f" % mse_prev)
    print("Model MSE: %f" % mse_model)

    if X_hat_ori is not None:
        # Calculate MSE for original model if provided
        mse_model_ori = np.mean((X_test[:, 1:] - X_hat_ori[:, 1:]) ** 2)
        print("Original Model MSE: %f" % mse_model_ori)

        return mse_prev, mse_model, mse_model_ori

    return mse_prev, mse_model


        
def compare_sequences(X_test, X_hat, X_hat_ori=None, save_results=None, gif=False, mse=True, n_sequences=3, nt=10):
    """
    Compare and display or save sequences from actual data, model predictions, and optionally original model predictions.

    Args:
        X_test (np.array): Actual test images.
        X_hat (np.array): Predicted images from the current model.
        X_hat_ori (np.array, optional): Predicted images from the original model implementation.
        save_results (str, optional): Directory path to save the comparison results. If not specified, results are not saved.
        gif (bool): Flag indicating whether to show the sequences as animated GIFs.
        mse (bool): Flag indicating whether to print mean squared error statistics.
        n_sequences (int): Number of sequences to display or save.
        nt (int): Number of time steps per sequence to display or save.

    Notes:
        Displays sequences using matplotlib for quick evaluation or saves them if a save path is provided.
    """
    if mse:
        evaluate_mse(X_test, X_hat, X_hat_ori)  # Evaluate and print MSE if enabled

    if save_results is not None and not os.path.exists(save_results):
        os.makedirs(save_results)  # Create save directory if it doesn't exist

    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]  # Calculate aspect ratio for image plotting
    plot_idx = np.random.permutation(X_test.shape[0])[:n_sequences]  # Select random sequences for comparison

    for seq_num, i in enumerate(plot_idx):
        n_plts = 2 if X_hat_ori is None else 3  # Determine the number of plot rows needed
        
        plt.figure(figsize=(nt*2, n_plts*2*aspect_ratio))
        gs = gridspec.GridSpec(n_plts, nt)
        gs.update(wspace=0., hspace=0.)

        if gif:
            print('Actual')
            visualize_sequence_as_gif((255 * X_test[i]).astype(np.uint8))  # Display actual sequence as GIF
            print('\nPredicted')
            visualize_sequence_as_gif((255 * X_hat[i]).astype(np.uint8))  # Display predicted sequence as GIF
            if X_hat_ori is not None:
                print('\nPredicted_Original')
                visualize_sequence_as_gif((255 * X_hat_ori[i]).astype(np.uint8))  # Display original predicted sequence as GIF

        for t in range(nt):
            plt.subplot(gs[0, t])
            plt.imshow(X_test[i, t], interpolation='none')
            plt.axis('off')
            if t == 0: plt.title('Actual', loc='center')

            plt.subplot(gs[1, t])
            plt.imshow(X_hat[i, t], interpolation='none')
            plt.axis('off')
            if t == 0: plt.title('Predicted', loc='center', y=0)

            if X_hat_ori is not None:
                plt.subplot(gs[2, t])
                plt.imshow(X_hat_ori[i, t], interpolation='none')
                plt.axis('off')
                if t == 0: plt.title('Predicted_Original', loc='center', y=0)

        if save_results is not None:
            save_path = os.path.join(save_results, f"sequence_{seq_num+1}.png")
            plt.savefig(save_path)
            plt.close() # Close the figure to avoid displaying it in Jupyter notebooks
            print(f"Saved: {save_path}")
        else:
            plt.show()
            plt.close()

def predict_future_sequence(prednet, X_test, start_idx, n_predictions):
    """
    Predict future frames using the PredNet model by iteratively updating with the last predicted frame.

    Args:
        prednet (model): Trained PredNet model to use for predictions.
        X_test (np.array): Array of test sequences.
        start_idx (int): Index at which to start predictions in the test dataset.
        n_predictions (int): Number of future frames to predict.

    Returns:
        tuple: Contains two arrays, one with predicted frames and one with corresponding actual frames for comparison.

    Raises:
        ValueError: If the start index does not allow for the required number of predictions.
    """
    # Ensure there is a subsequent sequence available for comparison
    if start_idx >= len(X_test) - 1:
        raise ValueError("No subsequent sequence available for comparison.")

    if n_predictions > 9:
        raise ValueError("Maximum number of predictions is 9.")
    # Get the initial sequence and the next sequence for comparison
    initial_sequence = np.expand_dims(X_test[start_idx], axis=0)
    initial_sequence_next = np.expand_dims(X_test[start_idx + 1], axis=0)

    # Start with the initial sequence to accumulate the predictions
    current_sequence = initial_sequence
    # Variable to store the ground truth frames corresponding to the predictions
    current_original_seq = initial_sequence

    for i in range(n_predictions):
        next_prediction = prednet(current_sequence)
        # Take the last frame from the prediction to extend the sequence
        last_frame_predicted = next_prediction[:, -1:, ...]

        # Concatenate the predicted frame to the sequence and use the last seq_length frames
        current_sequence = np.concatenate((current_sequence, last_frame_predicted), axis=1)
        # Update the ground truth sequence with the next frame
        current_original_seq = np.concatenate((current_original_seq, initial_sequence_next[:, i:i+1, ...]), axis=1)

    return next_prediction, current_original_seq

