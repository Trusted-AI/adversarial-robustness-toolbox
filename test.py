from config import DATA_PATH

import numpy as np
import os

import keras.backend as K
import tensorflow as tf

from src.classifiers import cnn
from src.utils import get_args, get_verbose_print, load_mnist

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
# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()

# Test on true train instances
scores = model.evaluate(X_train, Y_train, verbose=args.verbose)
v_print("\naccuracy on train: %.2f%%" % (scores[1] * 100))

# Test on true test instances
scores = model.evaluate(X_test, Y_test, verbose=args.verbose)
v_print("\naccuracy on test: %.2f%%" % (scores[1] * 100))

# get adversarial examples
ADV_PATH = os.path.join(DATA_PATH, "adversarial", "mnist", "fgsm", "cnn", "relu", "")

for eps in range(1, 11):
    X = np.load(ADV_PATH + "eps%.2f_train.npy" % (eps/10))

    scores = model.evaluate(X, Y_train, verbose=args.verbose)
    v_print("\naccuracy on train adversarial with  %.2f eps: %.2f%%" % (eps/10, scores[1] * 100))

    X = np.load(ADV_PATH + "eps%.2f_test.npy" % (eps/10))

    scores = model.evaluate(X, Y_test, verbose=args.verbose)
    v_print("\naccuracy on test adversarial with %.2f eps: %.2f%%" % (eps/10, scores[1] * 100))
