from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesses an image for model prediction.

    Args:
        image_path (str): The file path of the input image.

    Returns:
        numpy.ndarray: A preprocessed image as a NumPy array suitable for model prediction.
            The image is resized to (224, 224), normalized to [0, 1], and expanded with a batch dimension.
    """
    try:
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise Exception(f"Error during image preprocessing: {str(e)}")
