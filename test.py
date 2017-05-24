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
# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()
im_shape = X_train[0].shape

session = tf.Session()
K.set_session(session)

# Load classification model
MODEL_PATH = os.path.abspath(args.load)
model = cnn.load_model(MODEL_PATH, "best-weights.h5")

# Test
scores = model.evaluate(X_test, Y_test, verbose=args.verbose)
v_print("\naccuracy: %.2f%%" % (scores[1] * 100))
