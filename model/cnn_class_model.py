from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

def build_model(input_shape_3D):
    """
    Build and compile a convolutional neural network model for image classification.

    Parameters:
    - input_shape_3D (tuple): The desired shape of input images in 3D (height, width, channels).

    Returns:
    - model (keras.models.Sequential): The compiled convolutional neural network model.

    This function defines a convolutional neural network (CNN) model for image classification.
    The model architecture includes convolutional layers with max pooling, dropout layers,
    and fully connected layers. The model is compiled using the Adam optimizer and categorical
    crossentropy loss.

    Example:
    >>> input_shape = (256, 256, 3)
    >>> my_model = build_model(input_shape)
    """

    model = Sequential()

    model.add(Conv2D(32,(3,3),activation="relu",padding="same",input_shape=input_shape_3D))
    model.add(Conv2D(32,(3,3),activation="relu",padding="same"))
    model.add(MaxPooling2D(3,3))

    model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
    model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
    model.add(MaxPooling2D(3,3))

    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(MaxPooling2D(3,3))

    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))

    model.add(Conv2D(512,(5,5),activation="relu",padding="same"))
    model.add(Conv2D(512,(5,5),activation="relu",padding="same"))

    model.add(Flatten())

    model.add(Dense(1568,activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(38,activation="softmax"))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=['accuracy'])

    return model