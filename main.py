import numpy as np
import sys
from utils.data_loader import loadData
from utils.model_creator import make_or_restore_model
from utils.train import train
import itertools
import tensorflow as tf
from sklearn.metrics import precision_score, accuracy_score, recall_score


# Encoding variables
input_shape_3D = (256, 256, 3)
epochs = 50

# Loading Image Data as RGB Scale
train_gen, test_gen = loadData()

# Importing model # Build or restore model
model = make_or_restore_model(input_shape_3D)

# Asking the user if they want to train the model or not.
print("====================================================================================================")
train_model = str(input("Do you want to train the model? (Y/N): "))
print("====================================================================================================")
if ((train_model == "Y") or (train_model == "y")):
    train(model, train_gen, test_gen, epochs)
    model = make_or_restore_model(input_shape_3D)
elif ((train_model == "N") or (train_model == "n")):
    # Build or restore model
    model = make_or_restore_model(input_shape_3D)
    print("====================================================================================================")
    print("Model is loaded successfully")
    print("====================================================================================================")
else:
    print("Invalid Input. Run the program again.")
    sys.exit(1)

# Metrics
labels = []
predictions = []
for x,y in test_gen:
    labels.append(list(y.numpy()))
    predictions.append(tf.argmax(model.predict(x),1).numpy())

predictions = list(itertools.chain.from_iterable(predictions))
labels = list(itertools.chain.from_iterable(labels))

print("Test Accuracy   : {:.2f} %".format(accuracy_score(labels, predictions) * 100))
print("Precision Score : {:.2f} %".format(precision_score(labels, predictions, average='micro') * 100))
print("Recall Score    : {:.2f} %".format(recall_score(labels, predictions, average='micro') * 100))