from config import DATA_PATH

import json
import numpy as np
import os

import keras.backend as K
import tensorflow as tf

from src.classifiers import cnn
from src.utils import get_args, get_verbose_print, load_mnist, get_npy_files

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)

v_print = get_verbose_print(args.verbose)

comp_params = {"loss": 'categorical_crossentropy',
               "optimizer": 'adam',
               "metrics": ['accuracy']}

# --------------------------------------------------------------------------------------------- GET CLASSIFIER

session = tf.Session()
K.set_session(session)

# Load classification model
MODEL_PATH = os.path.abspath(args.load)
model = cnn.load_model(MODEL_PATH, "best-weights.h5")

# ------------------------------------------------------------------------------------------------------- TEST
results = {}

# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()

# Test on true train instances
scores = model.evaluate(X_train, Y_train, verbose=args.verbose)
v_print("\naccuracy on train: %.2f%%" % (scores[1] * 100))
results["train_accuracy"] = scores[1] * 100

# Test on true test instances
scores = model.evaluate(X_test, Y_test, verbose=args.verbose)
v_print("\naccuracy on test: %.2f%%" % (scores[1] * 100))
results["test_accuracy"] = scores[1] * 10

# get adversarial examples
ADV_PATH = os.path.join(DATA_PATH, "adversarial", "mnist")

for filepath in get_npy_files(ADV_PATH):

    X = np.load(filepath)
    Y = Y_train if "_train" in filepath else Y_test

    scores = model.evaluate(X, Y, verbose=args.verbose)
    v_print("\naccuracy on %s: %.2f%%" % (filepath, scores[1] * 100))
    results[filepath] = scores[1]*100

with open(os.path.join(MODEL_PATH, "accuracies.json"), "w") as json_file:
    json.dump(results, json_file)
