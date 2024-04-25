from model.cnn_model import build_model
import tensorflow as tf
import os, sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def make_or_restore_model(input_shape_3D=(256, 256, 3)):
    """
    Create a new model or restore an existing model from the latest checkpoint.

    Parameters:
    - input_shape_3D (tuple): The desired shape of input images in 3D (height, width, channels). Defaults to (224, 224, 3)

    Returns:
    - tf_model (tensorflow.keras.Model): The created or restored TensorFlow model.

    This function checks for the latest checkpoint in the "./ckpt/" directory. If a checkpoint
    is found, it restores the model from that checkpoint; otherwise, it creates a new model.

    Example:
    >>> input_shape = (256, 256, 3)
    >>> model = make_or_restore_model(input_shape)
    """

    latest_checkpoint = tf.train.latest_checkpoint("plant_disease/ckpt/")
    if latest_checkpoint is not None:
        print("====================================================================================================")
        print("Restoring Model from", latest_checkpoint)
        print("====================================================================================================")
        tf_model = build_model(input_shape_3D)
        checkpoint = tf.train.Checkpoint(model=tf_model)
        checkpoint.restore(latest_checkpoint)
        tf_model.summary()
        print("====================================================================================================")
        return tf_model
    else: 
        print("====================================================================================================")
        print("Error loading model from checkpoint!")
        print("====================================================================================================")
        return FileNotFoundError("Model Not Found!")