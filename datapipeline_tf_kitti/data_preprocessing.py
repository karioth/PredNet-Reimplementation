import tensorflow as tf

def preprocess_image(image, target_size):
    """
    Preprocesses an image by decoding it, resizing it, and normalizing its pixel values.

    Args:
        image (tf.Tensor): The input image tensor.
        target_size (tuple): The target size to resize the image to.

    Returns:
        tf.Tensor: The preprocessed image tensor.
    """
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)  # Resize images
    image = image / 255.0  # Normalize pixel values to [0, 1] ## or (image / 128.) - 1. to get [-1, 1]
    return image

def load_and_preprocess_image(path, target_size):
    """
    Loads an image from the given path and preprocesses it.

    Args:
        path (str): The path to the image file.
        target_size (tuple): The desired size of the image after preprocessing.

    Returns:
        The preprocessed image.
    """
    image = tf.io.read_file(path)
    return preprocess_image(image, target_size)