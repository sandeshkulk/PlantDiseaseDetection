import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.layers.experimental.preprocessing import Rescaling
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def loadData():
    """
    Load image data from the specified directory using TensorFlow's image_dataset_from_directory.

    Parameters:
    - input_shape_2D (tuple): The desired shape of input images in 2D (height, width).
    - seed (int): Seed for reproducibility when shuffling the dataset.

    Returns:
    - data (tf.data.Dataset): A TensorFlow dataset containing the loaded images and labels.

    This function uses TensorFlow's image_dataset_from_directory to load image data from the
    'rice_disease' directory. It returns a TensorFlow dataset object containing images and labels.

    Example:
    >>> input_shape = (256, 256)
    >>> seed_value = 42
    >>> dataset = loadData(input_shape, seed_value)
    """

    print("LOADING DATASET...")

    train_gen = image_dataset_from_directory(directory="dataset/train", image_size=(256, 256))
    test_gen = image_dataset_from_directory(directory="dataset/valid", image_size=(256, 256))

    rescale = Rescaling(scale=1.0/255)
    train_gen = train_gen.map(lambda image,label:(rescale(image),label))
    test_gen  = test_gen.map(lambda image,label:(rescale(image),label))
    return train_gen, test_gen