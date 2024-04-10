import tensorflow as tf
import os
import h5py
import numpy as np
import imageio
import IPython.display as display
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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



def visualize_sequence_as_gif(sequence):
    """
    Visualize a sequence of images (a video snippet) as an animated GIF with looping.
    """

    # Create an animated GIF with looping
    with imageio.get_writer('sequence.gif', mode='I', duration=0.3, loop=0) as writer:
        for image in sequence:
            writer.append_data(image)

    # Load the GIF and display it
    with open('sequence.gif', 'rb') as f:
        display.display(display.Image(data=f.read(), format='png'))
        # delete the gif file after displaying it
        os.remove('sequence.gif')


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
    for sequence in dataset.take(how_many):
        # Assuming the dataset yields batches of shape [batch_size, sequence_length, height, width, channels]
        first_sequence = (sequence[0][0].numpy() * 255).astype(np.uint8)  # Convert the first sequence to a NumPy array for visualization
        # to visualize the first sequence in the batch
        visualize_sequence_as_gif(first_sequence)
        # `first_sequence` shape is (sequence_length, image_height, image_width, channels)
        fig, axes = plt.subplots(1, sequence_length, figsize=(20, 2))
        for i, ax in enumerate(axes):
            ax.imshow(first_sequence[i].astype("uint8"))
            ax.axis('off')
        plt.show()
        
def evaluate_mse(X_test, X_hat, X_hat_ori = None):
    # Compare MSE of PredNet predictions vs. using last frame
    mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
    mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)
    
    print("Previous Frame MSE: %f" % mse_prev)
    print("Model MSE: %f" % mse_model)
    
    if X_hat_ori is not None:
        mse_model_ori = np.mean((X_test[:, 1:] - X_hat_ori[:, 1:]) ** 2)  
        print("Original Model MSE: %f" % mse_model_ori)
        
        return mse_prev, mse_model, mse_model_ori
    
    return mse_prev, mse_model

        
def compare_sequences(X_test, X_hat, X_hat_ori = None, save_results=None, gif=False, mse=True, n_sequences=3, nt=10):
    '''
    Display or save comparison of actual sequences and PredNet predictions.

    Parameters:
    - X_test: Actual pictures.
    - X_hat: Predicted pictures.
    - X_hat_ori: Predicted pictures by the original implementation.
    - save_results: Directory where results are saved. If no argument is passed, no saving occurs. 
    - gif: Flag if gif should be shown.
    - mse: Flag if mean squared errors should be printed.
    - n_sequences: Number of sequences to display or save. Default is 3.
    - nt: Number of timesteps per sequence. Default is 10.
    '''
    if mse:
        # Calculate MSE for model and previous frame
        evaluate_mse(X_test, X_hat, X_hat_ori)

    # Ensure the save directory exists
    if save_results is not None and not os.path.exists(save_results):
        os.makedirs(save_results)

    # Display or save some predictions
    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
    plot_idx = np.random.permutation(X_test.shape[0])[:n_sequences]

    for seq_num, i in enumerate(plot_idx):
        plt.figure(figsize=(nt*2, 4*aspect_ratio))
        
        if X_hat_ori is None:
            gs = gridspec.GridSpec(2, nt)
            gs.update(wspace=0., hspace=0.)
        else: 
            gs = gridspec.GridSpec(3, nt)
            gs.update(wspace=0., hspace=0.)

        if gif:
            # Ensure the sequence is in the correct format
            sequence_test = X_test[i]
            sequence_pred = X_hat[i]
            print('Actual')
            visualize_sequence_as_gif((255 * sequence_test).astype(np.uint8))
            print('\nPredicted')
            visualize_sequence_as_gif((255 * sequence_pred).astype(np.uint8))
            if X_hat_ori is not None:
                sequence_ori_pred = X_hat_ori[i]
                print('\nPredicted_Original')
                visualize_sequence_as_gif((255 * sequence_ori_pred).astype(np.uint8))

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
            plt.close()  # Close the figure to avoid displaying it in Jupyter notebooks
            print(f"Saved: {save_path}")
        else:
            plt.show()

def predict_future_sequence(prednet, X_test, start_idx, n_predictions):
    """
    Predicts future frames by iteratively updating the sequence with the last predicted frame,
    while also constructing the corresponding ground truth sequence.

    Parameters:
    - prednet: The trained PredNet model.
    - X_test: Test dataset containing sequences.
    - start_idx: Index to start the prediction from within the dataset.
    - n_predictions: The number of future frames to predict beyond the initial sequence.

    Returns:
    - A tuple of two arrays: one containing the predicted frames including images predicted on predicted images,
      and one containing the actual frames from the dataset for comparison.
    """

    # Ensure there is a subsequent sequence available for comparison
    if start_idx >= len(X_test) - 1:
        raise ValueError("No subsequent sequence available for comparison.")
    if n_predictions > 9:
        raise ValueError("Max number of predictions is 9")

    # Get the initial sequence and the next sequence for comparison
    initial_sequence = np.expand_dims(X_test[start_idx], axis=0)
    initial_sequence_next = np.expand_dims(X_test[start_idx + 1], axis=0)

    # Start with the initial sequence to accumulate the predictions
    current_sequence = initial_sequence

    # Variable to store the ground truth frames corresponding to the predictions
    current_original_seq = initial_sequence

    # Loop to predict each subsequent frame
    for i in range(n_predictions):
        # Predict the next set of frames using the current sequence
        next_prediction = prednet(current_sequence)

        # Take the last frame from the prediction to extend the sequence
        last_frame_predicted = next_prediction[:, -1:, ...]

        # Concatenate the predicted frame to the sequence and use the last seq_length frames
        current_sequence = np.concatenate((current_sequence, last_frame_predicted), axis=1)

        # Update the ground truth sequence with the next frame
        current_original_seq = np.concatenate((current_original_seq, initial_sequence_next[:, i:i+1, ...]), axis=1)
        #compare_sequences(current_original_seq, next_prediction, RESULTS_SAVE_DIR, save=False, n_sequences=1, nt=10+i)

    return next_prediction, current_original_seq
