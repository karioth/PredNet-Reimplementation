import os
import h5py
import tensorflow as tf
import numpy as np
import imageio
import IPython.display as display
import matplotlib.pyplot as plt



def read_hkl_file(train_hkl_path, test_hkl_path, val_hkl_path):
    '''Reads the sources from the hkl file and returns the unique sources for train, test and val sets.

    Args:
        train_hkl_path (str): The file path to the hkl file containing the training source data.
        test_hkl_path (str): The file path to the hkl file containing the testing source data.
        val_hkl_path (str): The file path to the hkl file containing the validation source data.

    Returns:
        tuple: A tuple containing the unique sources for the train, test, and val sets.
            train_sources (list): The unique sources for the train set.
            test_sources (list): The unique sources for the test set.
            val_sources (list): The unique sources for the val set.
    '''
    # Open the h5 file and read the sources
    for index, hkl_path in enumerate([train_hkl_path, test_hkl_path, val_hkl_path]):
        with h5py.File(hkl_path, 'r') as f:
            key = list(f.keys())[0]
            sources = f[key][:]
        # Assuming sources are byte strings, decode them to strings (if necessary)
        sources = [source.decode('utf-8') for source in sources]

        # Process each source to remove the category prefix
        cleaned_sources = [source.split('-')[1] for source in sources]

        # Remove duplicates by converting the list to a set
        unique_sources = set(cleaned_sources)

        # Optionally, convert back to a list if you need list functionality
        unique_sources = list(unique_sources)

        if index == 0:
            train_sources = unique_sources
        elif index == 1:
            test_sources = unique_sources
        else:
            val_sources = unique_sources
    return train_sources, test_sources, val_sources

def find_matching_folders(data_dir, target_folders):
    """
    Find matching folders in the given data directory that contain any of the target folders
    to filter the raw dataset to the relevant folders used in PredNet.

    Args:
        data_dir (str): The path to the data directory.
        target_folders (list): A list of target folder names.

    Returns:
        list: A list of matching folder paths.

    """
    matching_folder_paths = []

    # List all the directories in DATA_DIR
    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        # Check if it's a directory
        if os.path.isdir(entry_path):
            # Now check if this directory contains any of the target folders
            for target_folder in target_folders:
                target_folder_path = os.path.join(entry_path, target_folder, 'image_03', 'data')
                if os.path.exists(target_folder_path):
                    matching_folder_paths.append(target_folder_path)

    return matching_folder_paths

def get_relevant_paths(data_dir, hkl_paths):
    """
    Get the relevant paths for the train, test, and val sets.

    Args:
        data_dir (str): The path to the data directory.
        hkl_paths (list): A list of hkl file paths.

    Returns:
        tuple: A tuple containing the relevant paths for the train, test, and val sets.
            train_paths (list): The relevant paths for the train set.
            test_paths (list): The relevant paths for the test set.
            val_paths (list): The relevant paths for the val set.
    """
    # Get the unique sources for the train, test, and val sets
    train_sources, test_sources, val_sources = read_hkl_file(*hkl_paths)

    # Find the matching folders for the train, test, and val sets
    train_folders = find_matching_folders(data_dir, train_sources)
    test_folders = find_matching_folders(data_dir, test_sources)
    val_folders = find_matching_folders(data_dir, val_sources)

    return train_folders, test_folders, val_folders

def set_output_mode(batch_x, output_mode='error'):
    """
    Sets the output mode for the given batch of data.

    Args:
        batch_x: The input batch of data.
        output_mode: The desired output mode. Can be either 'error' or 'prediction'.

    Returns:
        A tuple containing the input batch and the corresponding prediction batch based on the specified output mode.
    """
    if output_mode == 'error':
        # For 'error' mode, batch_y is a dummy tensor of zeros since the actual error calculation is done elsewhere
        batch_y = tf.zeros((tf.shape(batch_x)[0],), dtype=tf.float32)
    elif output_mode == 'prediction':
        # For 'prediction' mode, batch_y is just batch_x (i.e., the model is expected to predict
        # the next frame based on previous frames).
        batch_y = batch_x
    # else: checked before
    #     raise ValueError("Unsupported output_mode. Choose either 'error' or 'prediction'.")
    return batch_x, batch_y

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
        print("Shape of the sequence:", first_sequence.shape)

        fig, axes = plt.subplots(1, sequence_length, figsize=(20, 2))
        for i, ax in enumerate(axes):
            ax.imshow(first_sequence[i].astype("uint8"))
            ax.axis('off')
        plt.show()