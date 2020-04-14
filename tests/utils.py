# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Module providing convenience functions specifically for unit tests.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import json
import time
import pickle
import unittest

import numpy as np

from art.utils import load_dataset

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------- TEST BASE CLASS
art_supported_frameworks = ["keras", "tensorflow", "pytorch", "scikitlearn"]


class TestBase(unittest.TestCase):
    """
    This class implements the base class for all unit tests.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(1234)

        cls.n_train = 1000
        cls.n_test = 100
        cls.batch_size = 16

        cls.create_image_dataset(n_train=cls.n_train, n_test=cls.n_test)

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris), _, _ = load_dataset("iris")

        cls.x_train_iris = x_train_iris
        cls.y_train_iris = y_train_iris
        cls.x_test_iris = x_test_iris
        cls.y_test_iris = y_test_iris

        cls._x_train_iris_original = cls.x_train_iris.copy()
        cls._y_train_iris_original = cls.y_train_iris.copy()
        cls._x_test_iris_original = cls.x_test_iris.copy()
        cls._y_test_iris_original = cls.y_test_iris.copy()

        import warnings

        # Filter warning for scipy, removed with scipy 1.4
        warnings.filterwarnings("ignore", ".*the output shape of zoom.*")

    @classmethod
    def create_image_dataset(cls, n_train, n_test):
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = load_dataset("mnist")
        cls.x_train_mnist = x_train_mnist[:n_train]
        cls.y_train_mnist = y_train_mnist[:n_train]
        cls.x_test_mnist = x_test_mnist[:n_test]
        cls.y_test_mnist = y_test_mnist[:n_test]

        cls._x_train_mnist_original = cls.x_train_mnist.copy()
        cls._y_train_mnist_original = cls.y_train_mnist.copy()
        cls._x_test_mnist_original = cls.x_test_mnist.copy()
        cls._y_test_mnist_original = cls.y_test_mnist.copy()

    def setUp(self):
        self.time_start = time.time()
        print("\n\n\n----------------------------------------------------------------------")

    def tearDown(self):
        time_end = time.time() - self.time_start
        test_name = ".".join(self.id().split(" ")[0].split(".")[-2:])
        logger.info("%s: completed in %.3f seconds" % (test_name, time_end))

        # Check that the test data has not been modified, only catches changes in attack.generate if self has been used
        np.testing.assert_array_almost_equal(
            self._x_train_mnist_original[0 : self.n_train], self.x_train_mnist, decimal=3
        )
        np.testing.assert_array_almost_equal(
            self._y_train_mnist_original[0 : self.n_train], self.y_train_mnist, decimal=3
        )
        np.testing.assert_array_almost_equal(self._x_test_mnist_original[0 : self.n_test], self.x_test_mnist, decimal=3)
        np.testing.assert_array_almost_equal(self._y_test_mnist_original[0 : self.n_test], self.y_test_mnist, decimal=3)

        np.testing.assert_array_almost_equal(self._x_train_iris_original, self.x_train_iris, decimal=3)
        np.testing.assert_array_almost_equal(self._y_train_iris_original, self.y_train_iris, decimal=3)
        np.testing.assert_array_almost_equal(self._x_test_iris_original, self.x_test_iris, decimal=3)
        np.testing.assert_array_almost_equal(self._y_test_iris_original, self.y_test_iris, decimal=3)


class ExpectedValue:
    def __init__(self, value, decimals):
        self.value = value
        self.decimals = decimals


# ----------------------------------------------------------------------------------------------- TEST MODELS FOR MNIST


def check_adverse_example_x(x_adv, x_original, max=1.0, min=0.0, bounded=True):
    """
    Performs basic checks on generated adversarial inputs (whether x_test or x_train)
    :param x_adv:
    :param x_original:
    :param max:
    :param min:
    :param bounded:
    :return:
    """
    assert bool((x_original == x_adv).all()) is False, "x_test_adv should have been different from x_test"

    if bounded:
        assert np.amax(x_adv) <= max, "x_test_adv values should have all been below {0}".format(max)
        assert np.amin(x_adv) >= min, "x_test_adv values should have all been above {0}".format(min)
    else:
        assert (x_adv > max).any(), "some x_test_adv values should have been above 1".format(max)
        assert (x_adv < min).any(), " some x_test_adv values should have all been below {0}".format(min)


def check_adverse_predicted_sample_y(y_pred_adv, y_non_adv):
    assert bool((y_non_adv == y_pred_adv).all()) is False, "Adverse predicted sample was not what was expected"


def is_valid_framework(framework):
    if framework not in art_supported_frameworks:
        raise Exception(
            "mlFramework value {0} is unsupported. Please use one of these valid values: {1}".format(
                framework, " ".join(art_supported_frameworks)
            )
        )
    return True


def _tf_weights_loader(dataset, weights_type, layer="DENSE", tf_version=1):
    filename = str(weights_type) + "_" + str(layer) + "_" + str(dataset) + ".npy"

    # pylint: disable=W0613
    # disable pylint because of API requirements for function
    if tf_version == 1:

        def _tf_initializer(_, dtype, partition_info):
            import tensorflow as tf

            weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", filename))
            return tf.constant(weights, dtype)

    elif tf_version == 2:

        def _tf_initializer(_, dtype):
            import tensorflow as tf

            weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", filename))
            return tf.constant(weights, dtype)

    else:
        raise ValueError("The TensorFlow version tf_version has to be either 1 or 2.")

    return _tf_initializer


def _kr_weights_loader(dataset, weights_type, layer="DENSE"):
    import keras.backend as k

    filename = str(weights_type) + "_" + str(layer) + "_" + str(dataset) + ".npy"

    def _kr_initializer(_, dtype=None):
        weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", filename))
        return k.variable(value=weights, dtype=dtype)

    return _kr_initializer


def _kr_tf_weights_loader(dataset, weights_type, layer="DENSE"):
    filename = str(weights_type) + "_" + str(layer) + "_" + str(dataset) + ".npy"
    weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", filename))
    return weights


def get_image_classifier_tf(from_logits=False, load_init=True, sess=None):
    import tensorflow as tf

    if tf.__version__[0] == "2":
        # sess is not required but set to None to return 2 values for v1 and v2
        classifier, sess = get_image_classifier_tf_v2(from_logits=from_logits), None
    else:
        classifier, sess = get_image_classifier_tf_v1(from_logits=from_logits, load_init=load_init, sess=sess)
    return classifier, sess


def get_image_classifier_tf_v1(from_logits=False, load_init=True, sess=None):
    """
    Standard TensorFlow classifier for unit testing.

    The following hyper-parameters were used to obtain the weights and biases:
    learning_rate: 0.01
    batch size: 10
    number of epochs: 2
    optimizer: tf.train.AdamOptimizer

    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`
    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :param sess: Computation session.
    :type sess: `tf.Session`
    :return: TensorFlowClassifier, tf.Session()
    """
    # pylint: disable=E0401
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if tf.__version__[0] == "2":
        import tensorflow.compat.v1 as tf

        tf.disable_eager_execution()
    from art.estimators.classification.tensorflow import TensorFlowClassifier

    # Define input and output placeholders
    input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    output_ph = tf.placeholder(tf.float32, shape=[None, 10])

    # Define the TensorFlow graph
    if load_init:
        conv = tf.layers.conv2d(
            input_ph,
            1,
            7,
            activation=tf.nn.relu,
            kernel_initializer=_tf_weights_loader("MNIST", "W", "CONV2D"),
            bias_initializer=_tf_weights_loader("MNIST", "B", "CONV2D"),
        )
    else:
        conv = tf.layers.conv2d(input_ph, 1, 7, activation=tf.nn.relu)

    conv = tf.layers.max_pooling2d(conv, 4, 4)
    flattened = tf.layers.flatten(conv)

    # Logits layer
    if load_init:
        logits = tf.layers.dense(
            flattened,
            10,
            kernel_initializer=_tf_weights_loader("MNIST", "W", "DENSE"),
            bias_initializer=_tf_weights_loader("MNIST", "B", "DENSE"),
        )
    else:
        logits = tf.layers.dense(flattened, 10)

    # probabilities
    probabilities = tf.keras.activations.softmax(x=logits)

    # Train operator
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    # TensorFlow session and initialization
    if sess is None:
        sess = tf.Session()
    elif not isinstance(sess, tf.Session):
        raise TypeError("An instance of `tf.Session` should be passed to `sess`.")

    sess.run(tf.global_variables_initializer())

    # Create the classifier
    if from_logits:
        tfc = TensorFlowClassifier(
            clip_values=(0, 1),
            input_ph=input_ph,
            output=logits,
            labels_ph=output_ph,
            train=train,
            loss=loss,
            learning=None,
            sess=sess,
        )
    else:
        tfc = TensorFlowClassifier(
            clip_values=(0, 1),
            input_ph=input_ph,
            output=probabilities,
            labels_ph=output_ph,
            train=train,
            loss=loss,
            learning=None,
            sess=sess,
        )

    return tfc, sess


def get_image_classifier_tf_v2(from_logits=False):
    """
    Standard TensorFlow v2 classifier for unit testing.

    The following hyper-parameters were used to obtain the weights and biases:
    learning_rate: 0.01
    batch size: 10
    number of epochs: 2
    optimizer: tf.train.AdamOptimizer

    :return: TensorFlowV2Classifier
    """
    # pylint: disable=E0401
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier

    if tf.__version__[0] != "2":
        raise ImportError("This function requires TensorFlow v2.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    model = Sequential()
    model.add(Conv2D(
        filters=1,
        kernel_size=7,
        activation="relu",
        kernel_initializer=_tf_weights_loader("MNIST", "W", "CONV2D", 2),
        bias_initializer=_tf_weights_loader("MNIST", "B", "CONV2D", 2),
        input_shape=(28, 28, 1),
    ))
    model.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding="valid", data_format=None))
    model.add(Flatten())
    if from_logits:
        model.add(Dense(
            10,
            activation="linear",
            kernel_initializer=_tf_weights_loader("MNIST", "W", "DENSE", 2),
            bias_initializer=_tf_weights_loader("MNIST", "B", "DENSE", 2),
        ))
    else:
        model.add(Dense(
            10,
            activation="softmax",
            kernel_initializer=_tf_weights_loader("MNIST", "W", "DENSE", 2),
            bias_initializer=_tf_weights_loader("MNIST", "B", "DENSE", 2),
        ))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss_object)

    # Create the classifier
    tfc = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=10,
        input_shape=(28, 28, 1),
        clip_values=(0, 1),
    )

    return tfc


def get_image_classifier_kr(
    loss_name="categorical_crossentropy", loss_type="function_losses", from_logits=False, load_init=True
):
    """
    Standard Keras classifier for unit testing

    The weights and biases are identical to the TensorFlow model in get_classifier_tf().

    :param loss_name: The name of the loss function.
    :type loss_name: `str`
    :param loss_type: The type of loss function definitions: label (loss function defined by string of its name),
                      function_losses (loss function imported from keras.losses), function_backend (loss function
                      imported from keras.backend)
    :type loss_type: `str`
    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`
    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :return: KerasClassifier, tf.Session()
    """
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    from art.estimators.classification.keras import KerasClassifier

    # Create simple CNN
    model = Sequential()

    if load_init:
        model.add(
            Conv2D(
                1,
                kernel_size=(7, 7),
                activation="relu",
                input_shape=(28, 28, 1),
                kernel_initializer=_kr_weights_loader("MNIST", "W", "CONV2D"),
                bias_initializer=_kr_weights_loader("MNIST", "B", "CONV2D"),
            )
        )
    else:
        model.add(Conv2D(1, kernel_size=(7, 7), activation="relu", input_shape=(28, 28, 1)))

    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())

    if from_logits:
        if load_init:
            model.add(
                Dense(
                    10,
                    activation="linear",
                    kernel_initializer=_kr_weights_loader("MNIST", "W", "DENSE"),
                    bias_initializer=_kr_weights_loader("MNIST", "B", "DENSE"),
                )
            )
        else:
            model.add(Dense(10, activation="linear"))
    else:
        if load_init:
            model.add(
                Dense(
                    10,
                    activation="softmax",
                    kernel_initializer=_kr_weights_loader("MNIST", "W", "DENSE"),
                    bias_initializer=_kr_weights_loader("MNIST", "B", "DENSE"),
                )
            )
        else:
            model.add(Dense(10, activation="softmax"))

    if loss_name == "categorical_hinge":
        if loss_type == "label":
            raise AttributeError("This combination of loss function options is not supported.")
        elif loss_type == "function_losses":
            loss = keras.losses.categorical_hinge
    elif loss_name == "categorical_crossentropy":
        if loss_type == "label":
            if from_logits:
                raise AttributeError("This combination of loss function options is not supported.")
            else:
                loss = loss_name
        elif loss_type == "function_losses":
            if from_logits:
                if int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3:

                    def categorical_crossentropy(y_true, y_pred):
                        return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

                    loss = categorical_crossentropy
                else:
                    raise AttributeError("This combination of loss function options is not supported.")
            else:
                loss = keras.losses.categorical_crossentropy
        elif loss_type == "function_backend":
            if from_logits:

                def categorical_crossentropy(y_true, y_pred):
                    return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = categorical_crossentropy
            else:
                loss = keras.backend.categorical_crossentropy
    elif loss_name == "sparse_categorical_crossentropy":
        if loss_type == "label":
            if from_logits:
                raise AttributeError("This combination of loss function options is not supported.")
            else:
                loss = loss_name
        elif loss_type == "function_losses":
            if from_logits:
                if int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3:

                    def sparse_categorical_crossentropy(y_true, y_pred):
                        return keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

                    loss = sparse_categorical_crossentropy
                else:
                    raise AttributeError("This combination of loss function options is not supported.")
            else:
                loss = keras.losses.sparse_categorical_crossentropy
        elif loss_type == "function_backend":
            if from_logits:

                def sparse_categorical_crossentropy(y_true, y_pred):
                    return keras.backend.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = sparse_categorical_crossentropy
            else:
                loss = keras.backend.sparse_categorical_crossentropy
    elif loss_name == "kullback_leibler_divergence":
        if loss_type == "label":
            raise AttributeError("This combination of loss function options is not supported.")
        elif loss_type == "function_losses":
            loss = keras.losses.kullback_leibler_divergence
        elif loss_type == "function_backend":
            raise AttributeError("This combination of loss function options is not supported.")
    elif loss_name == "cosine_similarity":
        if loss_type == "label":
            loss = loss_name
        elif loss_type == "function_losses":
            loss = keras.losses.cosine_similarity
        elif loss_type == "function_backend":
            loss = keras.backend.cosine_similarity

    else:
        raise ValueError("Loss name not recognised.")

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=from_logits)

    return krc


def get_image_classifier_kr_tf(loss_name="categorical_crossentropy", loss_type="function", from_logits=False):
    """
    Standard Keras classifier for unit testing

    The weights and biases are identical to the TensorFlow model in get_classifier_tf().

    :param loss_name: The name of the loss function.
    :type loss_name: `str`
    :param loss_type: The type of loss function definitions: label (loss function defined by string of its name),
                      function_losses (loss function), class (loss function generator)
    :type loss_type: `str`
    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`


    :return: KerasClassifier
    """
    # pylint: disable=E0401
    import tensorflow as tf

    if tf.__version__[0] == "2":
        tf.compat.v1.disable_eager_execution()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    from art.estimators.classification.keras import KerasClassifier

    # Create simple CNN
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(7, 7), activation="relu", input_shape=(28, 28, 1)))
    model.layers[-1].set_weights(
        [_kr_tf_weights_loader("MNIST", "W", "CONV2D"), _kr_tf_weights_loader("MNIST", "B", "CONV2D")]
    )
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())

    if from_logits:
        model.add(Dense(10, activation="linear"))
    else:
        model.add(Dense(10, activation="softmax"))

    model.layers[-1].set_weights(
        [_kr_tf_weights_loader("MNIST", "W", "DENSE"), _kr_tf_weights_loader("MNIST", "B", "DENSE")]
    )

    if loss_name == "categorical_hinge":
        if loss_type == "label":
            loss = loss_name
        elif loss_type == "function":
            loss = tf.keras.losses.categorical_hinge
        elif loss_type == "class":
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError("This combination of loss function options is not supported.")
            loss = tf.keras.losses.CategoricalHinge(reduction=reduction)
    elif loss_name == "categorical_crossentropy":
        if loss_type == "label":
            if from_logits:
                raise AttributeError
            else:
                loss = loss_name
        elif loss_type == "function":
            if from_logits:

                def categorical_crossentropy(y_true, y_pred):
                    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = categorical_crossentropy
            else:
                loss = tf.keras.losses.categorical_crossentropy
        elif loss_type == "class":
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError("This combination of loss function options is not supported.")
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, reduction=reduction)
    elif loss_name == "sparse_categorical_crossentropy":
        if loss_type == "label":
            if from_logits:
                raise AttributeError
            else:
                loss = loss_name
        elif loss_type == "function":
            if from_logits:

                def sparse_categorical_crossentropy(y_true, y_pred):
                    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = sparse_categorical_crossentropy
            else:
                loss = tf.keras.losses.sparse_categorical_crossentropy
        elif loss_type == "class":
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError("This combination of loss function options is not supported.")
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction=reduction)
    elif loss_name == "kullback_leibler_divergence":
        if loss_type == "label":
            loss = loss_name
        elif loss_type == "function":
            loss = tf.keras.losses.kullback_leibler_divergence
        elif loss_type == "class":
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError("This combination of loss function options is not supported.")
            loss = tf.keras.losses.KLDivergence(reduction=reduction)
    elif loss_name == "cosine_similarity":
        if loss_type == "label":
            loss = loss_name
        elif loss_type == "function":
            loss = tf.keras.losses.cosine_similarity
        elif loss_type == "class":
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError("This combination of loss function options is not supported.")
            loss = tf.keras.losses.CosineSimilarity(reduction=reduction)

    else:
        raise ValueError("Loss name not recognised.")

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=["accuracy"])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=from_logits)

    return krc


def get_image_classifier_kr_tf_binary():
    """
    Standard Tensorflow-Keras binary classifier for unit testing

    :return: KerasClassifier
    """
    # pylint: disable=E0401
    import tensorflow as tf

    if tf.__version__[0] == "2":
        tf.compat.v1.disable_eager_execution()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    from art.estimators.classification.keras import KerasClassifier

    # Create simple CNN
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(7, 7), activation="relu", input_shape=(28, 28, 1)))
    model.layers[-1].set_weights(
        [_kr_tf_weights_loader("MNIST_BINARY", "W", "CONV2D"), _kr_tf_weights_loader("MNIST_BINARY", "B", "CONV2D")]
    )
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.layers[-1].set_weights(
        [_kr_tf_weights_loader("MNIST_BINARY", "W", "DENSE"), _kr_tf_weights_loader("MNIST_BINARY", "B", "DENSE")]
    )

    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=["accuracy"])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False)

    return krc


def get_image_classifier_pt(from_logits=False, load_init=True):
    """
    Standard PyTorch classifier for unit testing.

    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`
    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :return: PyTorchClassifier
    """
    import torch

    from art.estimators.classification.pytorch import PyTorchClassifier

    class Model(torch.nn.Module):
        """
        Create model for pytorch.

        The weights and biases are identical to the TensorFlow model in get_classifier_tf().
        """

        def __init__(self):
            super(Model, self).__init__()

            self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
            self.pool = torch.nn.MaxPool2d(4, 4)
            self.fullyconnected = torch.nn.Linear(25, 10)

            if load_init:
                w_conv2d = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "W_CONV2D_MNIST.npy")
                )
                b_conv2d = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "B_CONV2D_MNIST.npy")
                )
                w_dense = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "W_DENSE_MNIST.npy")
                )
                b_dense = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "B_DENSE_MNIST.npy")
                )

                w_conv2d_pt = w_conv2d.reshape((1, 1, 7, 7))

                self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d_pt))
                self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
                self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(np.transpose(w_dense)))
                self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

        # pylint: disable=W0221
        # disable pylint because of API requirements for function
        def forward(self, x):
            """
            Forward function to evaluate the model
            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.conv(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            x = x.reshape(-1, 25)
            x = self.fullyconnected(x)
            if not from_logits:
                x = torch.nn.functional.softmax(x, dim=1)
            return x

    # Define the network
    model = Model()

    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Get classifier
    ptc = PyTorchClassifier(
        model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
    )

    return ptc


def get_classifier_bb(defences=None):
    """
    Standard BlackBox classifier for unit testing

    :return: BlackBoxClassifier
    """
    from art.estimators.classification.blackbox import BlackBoxClassifier
    from art.utils import to_categorical

    # define blackbox classifier
    def predict(x):
        with open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/mnist", "api_output.txt")
        ) as json_file:
            predictions = json.load(json_file)
        return to_categorical(predictions["values"][: len(x)], nb_classes=10)

    bbc = BlackBoxClassifier(predict, (28, 28, 1), 10, clip_values=(0, 255), preprocessing_defences=defences)
    return bbc


def get_classifier_mx():
    """
    Standard MXNet classifier for unit testing

    :return: MXNetClassifier
    """
    import mxnet
    from mxnet.gluon.nn import Conv2D, MaxPool2D, Flatten, Dense
    from art.estimators.classification import MXClassifier

    model = mxnet.gluon.nn.Sequential()
    with model.name_scope():
        model.add(
            Conv2D(channels=1, kernel_size=7, activation="relu"),
            MaxPool2D(pool_size=4, strides=4),
            Flatten(),
            Dense(10),
        )
    model.initialize(init=mxnet.init.Xavier())

    # Create optimizer
    loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mxnet.gluon.Trainer(model.collect_params(), "sgd", {"learning_rate": 0.1})

    # # Fit classifier
    # classifier = MXClassifier(model=net, loss=loss, clip_values=(0, 1), input_shape=(1, 28, 28), nb_classes=10,
    #                           optimizer=trainer)
    # classifier.fit(x_train, y_train, batch_size=128, nb_epochs=2)
    # cls.classifier = classifier

    # Get classifier
    mxc = MXClassifier(
        model=model,
        loss=loss,
        input_shape=(28, 28, 1),
        nb_classes=10,
        optimizer=trainer,
        ctx=None,
        channel_index=1,
        clip_values=(0, 1),
        defences=None,
        preprocessing=(0, 1),
    )

    return mxc


# ------------------------------------------------------------------------------------------------ TEST MODELS FOR IRIS


def get_tabular_classifier_tf(load_init=True, sess=None):
    import tensorflow as tf

    if tf.__version__[0] == "2":
        # sess is not required but set to None to return 2 values for v1 and v2
        classifier, sess = get_tabular_classifier_tf_v2(), None
    else:
        classifier, sess = get_tabular_classifier_tf_v1(load_init=load_init, sess=sess)
    return classifier, sess


def get_tabular_classifier_tf_v1(load_init=True, sess=None):
    """
    Standard TensorFlow classifier for unit testing.

    The following hyper-parameters were used to obtain the weights and biases:

    * learning_rate: 0.01
    * batch size: 5
    * number of epochs: 200
    * optimizer: tf.train.AdamOptimizer

    The model is trained of 70% of the dataset, and 30% of the training set is used as validation split.

    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :param sess: Computation session.
    :type sess: `tf.Session`
    :return: The trained model for Iris dataset and the session.
    :rtype: `tuple(TensorFlowClassifier, tf.Session)`
    """
    import tensorflow as tf

    if tf.__version__[0] == "2":
        # pylint: disable=E0401
        import tensorflow.compat.v1 as tf

        tf.disable_eager_execution()
    from art.estimators.classification.tensorflow import TensorFlowClassifier

    # Define input and output placeholders
    input_ph = tf.placeholder(tf.float32, shape=[None, 4])
    output_ph = tf.placeholder(tf.int32, shape=[None, 3])

    # Define the TensorFlow graph
    if load_init:
        dense1 = tf.layers.dense(
            input_ph,
            10,
            kernel_initializer=_tf_weights_loader("IRIS", "W", "DENSE1"),
            bias_initializer=_tf_weights_loader("IRIS", "B", "DENSE1"),
        )
        dense2 = tf.layers.dense(
            dense1,
            10,
            kernel_initializer=_tf_weights_loader("IRIS", "W", "DENSE2"),
            bias_initializer=_tf_weights_loader("IRIS", "B", "DENSE2"),
        )
        logits = tf.layers.dense(
            dense2,
            3,
            kernel_initializer=_tf_weights_loader("IRIS", "W", "DENSE3"),
            bias_initializer=_tf_weights_loader("IRIS", "B", "DENSE3"),
        )
    else:
        dense1 = tf.layers.dense(input_ph, 10)
        dense2 = tf.layers.dense(dense1, 10)
        logits = tf.layers.dense(dense2, 3)

    # Train operator
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    # TensorFlow session and initialization
    if sess is None:
        sess = tf.Session()
    elif not isinstance(sess, tf.Session):
        raise TypeError("An instance of `tf.Session` should be passed to `sess`.")

    sess.run(tf.global_variables_initializer())

    # Train the classifier
    tfc = TensorFlowClassifier(
        clip_values=(0, 1),
        input_ph=input_ph,
        output=logits,
        labels_ph=output_ph,
        train=train,
        loss=loss,
        learning=None,
        sess=sess,
        channel_index=1,
    )

    return tfc, sess


def get_tabular_classifier_tf_v2():
    """
    Standard TensorFlow v2 classifier for unit testing.

    The following hyper-parameters were used to obtain the weights and biases:

    * learning_rate: 0.01
    * batch size: 5
    * number of epochs: 200
    * optimizer: tf.train.AdamOptimizer

    The model is trained of 70% of the dataset, and 30% of the training set is used as validation split.

    :return: The trained model for Iris dataset and the session.
    :rtype: `TensorFlowV2Classifier`
    """
    # pylint: disable=E0401
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier

    if tf.__version__[0] != "2":
        raise ImportError("This function requires TensorFlow v2.")

    class TensorFlowModel(Model):
        """
        Standard TensorFlow model for unit testing
        """

        def __init__(self):
            super(TensorFlowModel, self).__init__()
            self.dense1 = Dense(
                10,
                activation="linear",
                kernel_initializer=_tf_weights_loader("IRIS", "W", "DENSE1", tf_version=2),
                bias_initializer=_tf_weights_loader("IRIS", "B", "DENSE1", tf_version=2),
            )
            self.dense2 = Dense(
                10,
                activation="linear",
                kernel_initializer=_tf_weights_loader("IRIS", "W", "DENSE2", tf_version=2),
                bias_initializer=_tf_weights_loader("IRIS", "B", "DENSE2", tf_version=2),
            )
            self.logits = Dense(
                3,
                activation="linear",
                kernel_initializer=_tf_weights_loader("IRIS", "W", "DENSE3", tf_version=2),
                bias_initializer=_tf_weights_loader("IRIS", "B", "DENSE3", tf_version=2),
            )

        def call(self, x):
            """
            Call function to evaluate the model

            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.logits(x)
            return x

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    model = TensorFlowModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Create the classifier
    tfc = TensorFlowV2Classifier(
        model=model, loss_object=loss_object, train_step=train_step, nb_classes=3, input_shape=(4,), clip_values=(0, 1)
    )

    return tfc


def get_tabular_classifier_scikit_list(clipped=False):
    model_list_names = [
        "decisionTreeClassifier",
        "extraTreeClassifier",
        "adaBoostClassifier",
        "baggingClassifier",
        "extraTreesClassifier",
        "gradientBoostingClassifier",
        "randomForestClassifier",
        "logisticRegression",
        "svc",
        "linearSVC",
    ]
    if clipped:
        classifier_list = [
            # os.path.join(os.path.dirname(os.path.dirname(__file__)),'resources/models', 'W_DENSE3_IRIS.npy')
            pickle.load(
                open(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "resources/models/scikit/",
                        model_name + "iris_clipped.sav",
                    ),
                    "rb",
                )
            )
            for model_name in model_list_names
        ]
    else:
        classifier_list = [
            pickle.load(
                open(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "resources/models/scikit/",
                        model_name + "iris_unclipped.sav",
                    ),
                    "rb",
                )
            )
            for model_name in model_list_names
        ]

    return classifier_list


def get_tabular_classifier_kr(load_init=True):
    """
    Standard Keras classifier for unit testing on Iris dataset. The weights and biases are identical to the TensorFlow
    model in `get_iris_classifier_tf`.

    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :return: The trained model for Iris dataset and the session.
    :rtype: `tuple(KerasClassifier, tf.Session)`
    """
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    from art.estimators.classification.keras import KerasClassifier

    # Create simple CNN
    model = Sequential()

    if load_init:
        model.add(
            Dense(
                10,
                input_shape=(4,),
                activation="relu",
                kernel_initializer=_kr_weights_loader("IRIS", "W", "DENSE1"),
                bias_initializer=_kr_weights_loader("IRIS", "B", "DENSE1"),
            )
        )
        model.add(
            Dense(
                10,
                activation="relu",
                kernel_initializer=_kr_weights_loader("IRIS", "W", "DENSE2"),
                bias_initializer=_kr_weights_loader("IRIS", "B", "DENSE2"),
            )
        )
        model.add(
            Dense(
                3,
                activation="softmax",
                kernel_initializer=_kr_weights_loader("IRIS", "W", "DENSE3"),
                bias_initializer=_kr_weights_loader("IRIS", "B", "DENSE3"),
            )
        )
    else:
        model.add(Dense(10, input_shape=(4,), activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(3, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False, channel_index=1)

    return krc


def get_tabular_classifier_pt(load_init=True):
    """
    Standard PyTorch classifier for unit testing on Iris dataset.

    :param load_init: Load the initial weights if True.
    :type load_init: `bool`
    :return: Trained model for Iris dataset.
    :rtype: :class:`.PyTorchClassifier`
    """
    import torch

    from art.estimators.classification.pytorch import PyTorchClassifier

    class Model(torch.nn.Module):
        """
        Create Iris model for PyTorch.

        The weights and biases are identical to the TensorFlow model in `get_iris_classifier_tf`.
        """

        def __init__(self):
            super(Model, self).__init__()

            self.fully_connected1 = torch.nn.Linear(4, 10)
            self.fully_connected2 = torch.nn.Linear(10, 10)
            self.fully_connected3 = torch.nn.Linear(10, 3)

            if load_init:
                w_dense1 = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "W_DENSE1_IRIS.npy")
                )
                b_dense1 = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "B_DENSE1_IRIS.npy")
                )
                w_dense2 = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "W_DENSE2_IRIS.npy")
                )
                b_dense2 = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "B_DENSE2_IRIS.npy")
                )
                w_dense3 = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "W_DENSE3_IRIS.npy")
                )
                b_dense3 = np.load(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models", "B_DENSE3_IRIS.npy")
                )

                self.fully_connected1.weight = torch.nn.Parameter(torch.Tensor(np.transpose(w_dense1)))
                self.fully_connected1.bias = torch.nn.Parameter(torch.Tensor(b_dense1))
                self.fully_connected2.weight = torch.nn.Parameter(torch.Tensor(np.transpose(w_dense2)))
                self.fully_connected2.bias = torch.nn.Parameter(torch.Tensor(b_dense2))
                self.fully_connected3.weight = torch.nn.Parameter(torch.Tensor(np.transpose(w_dense3)))
                self.fully_connected3.bias = torch.nn.Parameter(torch.Tensor(b_dense3))

        # pylint: disable=W0221
        # disable pylint because of API requirements for function
        def forward(self, x):
            x = self.fully_connected1(x)
            x = self.fully_connected2(x)
            logit_output = self.fully_connected3(x)

            return logit_output

    # Define the network
    model = Model()

    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Get classifier
    ptc = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(4,),
        nb_classes=3,
        clip_values=(0, 1),
        channel_index=1,
    )

    return ptc


# -------------------------------------------------------------------------------------------- RANDOM NUMBER GENERATORS


def master_seed(seed=1234, set_random=True, set_numpy=True, set_tensorflow=False, set_mxnet=False, set_torch=False):
    """
    Set the seed for all random number generators used in the library. This ensures experiments reproducibility and
    stable testing.

    :param seed: The value to be seeded in the random number generators.
    :type seed: `int`
    :param set_random: The flag to set seed for `random`.
    :type set_random: `bool`
    :param set_numpy: The flag to set seed for `numpy`.
    :type set_numpy: `bool`
    :param set_tensorflow: The flag to set seed for `tensorflow`.
    :type set_tensorflow: `bool`
    :param set_mxnet: The flag to set seed for `mxnet`.
    :type set_mxnet: `bool`
    :param set_torch: The flag to set seed for `torch`.
    :type set_torch: `bool`
    """
    import numbers

    if not isinstance(seed, numbers.Integral):
        raise TypeError("The seed for random number generators has to be an integer.")

    # Set Python seed
    if set_random:
        import random

        random.seed(seed)

    # Set Numpy seed
    if set_numpy:
        np.random.seed(seed)
        np.random.RandomState(seed)

    # Now try to set seed for all specific frameworks
    if set_tensorflow:
        try:
            import tensorflow as tf

            logger.info("Setting random seed for TensorFlow.")
            if tf.__version__[0] == "2":
                tf.random.set_seed(seed)
            else:
                tf.set_random_seed(seed)
        except ImportError:
            logger.info("Could not set random seed for TensorFlow.")

    if set_mxnet:
        try:
            import mxnet as mx

            logger.info("Setting random seed for MXNet.")
            mx.random.seed(seed)
        except ImportError:
            logger.info("Could not set random seed for MXNet.")

    if set_torch:
        try:
            logger.info("Setting random seed for PyTorch.")
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            logger.info("Could not set random seed for PyTorch.")
