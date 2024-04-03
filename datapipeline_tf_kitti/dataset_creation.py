import tensorflow as tf
import os
from datapipeline_tf_kitti.dataset_utils import set_output_mode
from datapipeline_tf_kitti.data_preprocessing import load_and_preprocess_image


def make_dataset_for_folder(folder_path, sequence_length, target_size, sequence_start_mode):
    """
    Creates a dataset of image sequences from a folder of image files.

    Args:
        folder_path (str): The path to the folder containing the image files.
        sequence_length (int): The length of each image sequence.
        target_size (tuple): The target size of the images after preprocessing.
        sequence_start_mode (str): The mode for determining the start of each sequence. 
            Possible values are 'all' (start a new sequence at every image) or 'unique' 
            (start a new sequence at every sequence_length-th image).

    Returns:
        tf.data.Dataset: A dataset of image sequences.

    """
    # Check if the folder is empty; if it is, return None or an empty dataset -- due to problems with google drive this could happen
    if not os.listdir(folder_path):
        print(f"Skipping empty folder: {folder_path}")
        return tf.data.Dataset.from_tensor_slices([])
    
    # List and sort image files
    filenames = sorted(
        [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.split('.')[0].isdigit()],
        key=lambda f: int(os.path.basename(f).split('.')[0])
    )

    # Create a dataset of image file paths
    path_dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Load and preprocess images
    image_dataset = path_dataset.map(lambda x: load_and_preprocess_image(x, target_size), num_parallel_calls=tf.data.AUTOTUNE)

    # Create sequences
    if sequence_start_mode == 'all':
        sequence_dataset = image_dataset.window(size=sequence_length, shift=1, drop_remainder=True)
    elif sequence_start_mode == 'unique':
        sequence_dataset = image_dataset.window(size=sequence_length, shift=sequence_length, drop_remainder=True)
    sequence_dataset = sequence_dataset.flat_map(lambda x: x.batch(sequence_length))

    return sequence_dataset

def make_dataset(folder_paths, sequence_length, batch_size, target_size, shuffle=False, shuffle_buffer_size=100, sequence_start_mode='all', output_mode='error', N_seq=None):
    """
    Create a dataset from the given folder paths.

    Args:
        folder_paths (list): List of folder paths containing video frames.
        sequence_length (int): Length of each sequence.
        batch_size (int): Number of sequences per batch.
        target_size (tuple): Target size of the video frames.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        shuffle_buffer_size (int, optional): Buffer size for shuffling. Defaults to 100.
        sequence_start_mode (str, optional): Sequence start mode. Must be either 'all' or 'unique'. Defaults to 'all'.
        output_mode (str, optional): Output mode. Must be either 'error' or 'prediction'. Defaults to 'error'.
        N_seq (int, optional): Maximum number of sequences to include in the dataset. Defaults to None.

    Returns:
        tf.data.Dataset: The created dataset.
    """
    
    # Ensure valid sequence_start_mode and output_mode values
    assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
    assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
    # Initialize an empty dataset
    all_videos_dataset = tf.data.Dataset.from_tensor_slices([])

    # Create a dataset for each folder and concatenate
    for folder_path in folder_paths:
        folder_dataset = make_dataset_for_folder(folder_path, sequence_length, target_size, sequence_start_mode)

        # Check if folder_dataset is not None or not empty before concatenating
        if folder_dataset.cardinality().numpy() != 0:
            all_videos_dataset = all_videos_dataset.concatenate(folder_dataset)
    # Shuffle the dataset if needed
    if shuffle:
        all_videos_dataset = all_videos_dataset.shuffle(buffer_size=shuffle_buffer_size)  # Adjust buffer_size based on dataset size

    # Apply N_seq limit
    if N_seq is not None:
        all_videos_dataset = all_videos_dataset.take(N_seq)

    # Batch sequences across all videos
    batched_dataset = all_videos_dataset.batch(batch_size, drop_remainder=True)

    
    # Map the set_output_mode function over the dataset
    # i think this could be done differently but this way it works with the existing code
    batched_dataset = batched_dataset.map(lambda batch_x: set_output_mode(batch_x, output_mode=output_mode))

    def set_shape(batch_x, batch_y):
        batch_x.set_shape([None, sequence_length, *target_size, 3])
        batch_y.set_shape([None])  # Assuming batch_y is a 1D tensor with the batch size dimension only
        return batch_x, batch_y
    # After batching dataset, apply the shape setting function
    batched_dataset = batched_dataset.map(set_shape)

    # repeat the dataset indefinitely
    batched_dataset = batched_dataset.repeat()

    batched_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)

    return batched_dataset