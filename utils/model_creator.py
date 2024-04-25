from model.cnn_class_model import build_model
import tensorflow as tf
import os

def make_or_restore_model(input_shape_3D):
    """
    Create a new model or restore an existing model from the latest checkpoint.

    Parameters:
    - input_shape_3D (tuple): The desired shape of input images in 3D (height, width, channels).

    Returns:
    - tf_model (tensorflow.keras.Model): The created or restored TensorFlow model.

    This function checks for the latest checkpoint in the "./ckpt/" directory. If a checkpoint
    is found, it restores the model from that checkpoint; otherwise, it creates a new model.

    Example:
    >>> input_shape = (224, 224, 3)
    >>> model = make_or_restore_model(input_shape)
    """

    latest_checkpoint = tf.train.latest_checkpoint("ckpt/")
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
        print("Creating a new model")
        print("====================================================================================================")
        tf_model = build_model(input_shape_3D)
        tf_model.summary()
        print("====================================================================================================")
        return tf_model


def save_model_checkpoint(tf_model, epoch):
    """
    Save the model checkpoint.

    Parameters:
    - tf_model (tensorflow.keras.Model): The TensorFlow model to save.
    - epoch (int): The epoch number to include in the checkpoint filename.

    Returns:
    None

    This function saves the model checkpoint with the specified epoch number in the "./ckpt/" directory.

    Example:
    >>> save_model_checkpoint(my_model, epoch=5)
    """

    checkpoint_prefix = os.path.join("ckpt/model-ckpt")
    checkpoint = tf.train.Checkpoint(model=tf_model)
    checkpoint.save(file_prefix=checkpoint_prefix + "-{}".format(epoch))
