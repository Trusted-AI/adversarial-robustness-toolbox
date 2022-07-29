# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from art.estimators.classification.keras import KerasClassifier
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowClassifier
from art.metrics.metrics import empirical_robustness, clever_t, clever_u, clever, loss_sensitivity, wasserstein_distance
from art.utils import load_mnist

from tests.utils import master_seed

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100

R_L1 = 40
R_L2 = 2
R_LI = 0.1


class TestMetrics(unittest.TestCase):
    def setUp(self):
        master_seed(seed=42)

    def test_emp_robustness_mnist(self):
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

        # Get classifier
        classifier = self._cnn_mnist_k([28, 28, 1])
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2, verbose=0)

        # Compute minimal perturbations
        params = {"eps_step": 1.0, "eps": 1.0}
        emp_robust = empirical_robustness(classifier, x_train, str("fgsm"), params)
        self.assertAlmostEqual(emp_robust, 1.000369094488189, 3)

        params = {"eps_step": 0.1, "eps": 0.2}
        emp_robust = empirical_robustness(classifier, x_train, str("fgsm"), params)
        self.assertLessEqual(emp_robust, 0.65)

    def test_loss_sensitivity(self):
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

        # Get classifier
        classifier = self._cnn_mnist_k([28, 28, 1])
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2, verbose=0)

        sensitivity = loss_sensitivity(classifier, x_train, y_train)
        self.assertGreaterEqual(sensitivity, 0)

    # def testNearestNeighborDist(self):
    #     # Get MNIST
    #     (x_train, y_train), (_, _), _, _ = load_mnist()
    #     x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
    #
    #     # Get classifier
    #     classifier = self._cnn_mnist_k([28, 28, 1])
    #     classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)
    #
    #     dist = nearest_neighbour_dist(classifier, x_train, x_train, str('fgsm'))
    #     self.assertGreaterEqual(dist, 0)

    @staticmethod
    def _cnn_mnist_k(input_shape):
        import tensorflow as tf

        tf_version = [int(v) for v in tf.__version__.split(".")]
        if tf_version[0] == 2 and tf_version[1] >= 3:
            tf.compat.v1.disable_eager_execution()
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        else:
            import keras
            from keras.models import Sequential
            from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))

        model.compile(
            loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
        )

        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        return classifier

    @staticmethod
    def _create_tfclassifier():
        """
        To create a simple TensorFlowClassifier for testing.
        :return:
        """
        import tensorflow as tf

        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        labels_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the TensorFlow graph
        conv = tf.layers.conv2d(input_ph, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # TensorFlow session and initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Create the classifier
        tfc = TensorFlowClassifier(
            input_ph=input_ph,
            output=logits,
            labels_ph=labels_ph,
            train=train,
            loss=loss,
            learning=None,
            sess=sess,
            clip_values=(0, 1),
        )

        return tfc

    @staticmethod
    def _create_krclassifier():
        """
        To create a simple KerasClassifier for testing.
        :return:
        """
        import tensorflow as tf

        tf_version = [int(v) for v in tf.__version__.split(".")]
        if tf_version[0] == 2 and tf_version[1] >= 3:
            tf.compat.v1.disable_eager_execution()
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        else:
            import keras
            from keras.models import Sequential
            from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))

        model.compile(
            loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
        )

        # Get the classifier
        krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

        return krc

    @staticmethod
    def _create_ptclassifier():
        """
        To create a simple PyTorchClassifier for testing.
        :return:
        """

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(1, 16, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc = nn.Linear(2304, 10)

            def forward(self, x):
                x = self.pool(f.relu(self.conv(x)))
                x = x.view(-1, 2304)
                logit_output = self.fc(x)

                return logit_output

        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Get classifier
        ptc = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
        )

        return ptc

    @unittest.skipIf(tf.__version__[0] == "2", reason="Skip unittests for TensorFlow v2.")
    def test_2_clever_tf(self):
        """
        Test with TensorFlow.
        :return:
        """
        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Get the classifier
        tfc = self._create_tfclassifier()
        tfc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=1)

        # TODO Need to configure r
        # Test targeted clever
        res0 = clever_t(tfc, x_test[-1], 2, 10, 5, R_L1, norm=1, pool_factor=3)
        res1 = clever_t(tfc, x_test[-1], 2, 10, 5, R_L2, norm=2, pool_factor=3)
        res2 = clever_t(tfc, x_test[-1], 2, 10, 5, R_LI, norm=np.inf, pool_factor=3)
        logger.info("Targeted TensorFlow: %f %f %f", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)

        # Test untargeted clever
        res0 = clever_u(tfc, x_test[-1], 10, 5, R_L1, norm=1, pool_factor=3, verbose=False)
        res1 = clever_u(tfc, x_test[-1], 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
        res2 = clever_u(tfc, x_test[-1], 10, 5, R_LI, norm=np.inf, pool_factor=3, verbose=False)
        logger.info("Untargeted TensorFlow: %f %f %f", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)

    def test_clever_kr(self):
        """
        Test with keras.
        :return:
        """
        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Get the classifier
        krc = self._create_krclassifier()
        krc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=1, verbose=0)

        # Test targeted clever
        res0 = clever_t(krc, x_test[-1], 2, 10, 5, R_L1, norm=1, pool_factor=3)
        res1 = clever_t(krc, x_test[-1], 2, 10, 5, R_L2, norm=2, pool_factor=3)
        res2 = clever_t(krc, x_test[-1], 2, 10, 5, R_LI, norm=np.inf, pool_factor=3)
        logger.info("Targeted Keras: %f %f %f", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)

        # Test untargeted clever
        res0 = clever_u(krc, x_test[-1], 10, 5, R_L1, norm=1, pool_factor=3, verbose=False)
        res1 = clever_u(krc, x_test[-1], 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
        res2 = clever_u(krc, x_test[-1], 10, 5, R_LI, norm=np.inf, pool_factor=3, verbose=False)
        logger.info("Untargeted Keras: %f %f %f", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)

    def test_3_clever_pt(self):
        """
        Test with pytorch.
        :return:
        """
        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]
        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

        # Get the classifier
        ptc = self._create_ptclassifier()
        ptc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=1)

        # Test targeted clever
        res0 = clever_t(ptc, x_test[-1], 2, 10, 5, R_L1, norm=1, pool_factor=3)
        res1 = clever_t(ptc, x_test[-1], 2, 10, 5, R_L2, norm=2, pool_factor=3)
        res2 = clever_t(ptc, x_test[-1], 2, 10, 5, R_LI, norm=np.inf, pool_factor=3)
        logger.info("Targeted PyTorch: %f %f %f", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)

        # Test untargeted clever
        res0 = clever_u(ptc, x_test[-1], 10, 5, R_L1, norm=1, pool_factor=3, verbose=False)
        res1 = clever_u(ptc, x_test[-1], 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
        res2 = clever_u(ptc, x_test[-1], 10, 5, R_LI, norm=np.inf, pool_factor=3, verbose=False)
        logger.info("Untargeted PyTorch: %f %f %f", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)

    def test_clever_l2_no_target(self):
        batch_size = 100
        (x_train, y_train), (x_test, _), _, _ = load_mnist()

        # Get the classifier
        krc = self._create_krclassifier()
        krc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=2, verbose=0)

        scores = clever(krc, x_test[0], 5, 5, 3, 2, target=None, c_init=1, pool_factor=10, verbose=False)
        logger.info("Clever scores for n-1 classes: %s %s", str(scores), str(scores.shape))
        self.assertEqual(scores.shape, (krc.nb_classes - 1,))

    def test_clever_l2_no_target_sorted(self):
        batch_size = 100
        (x_train, y_train), (x_test, _), _, _ = load_mnist()

        # Get the classifier
        krc = self._create_krclassifier()
        krc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=2, verbose=0)

        scores = clever(
            krc, x_test[0], 5, 5, 3, 2, target=None, target_sort=True, c_init=1, pool_factor=10, verbose=False
        )
        logger.info("Clever scores for n-1 classes: %s %s", str(scores), str(scores.shape))
        # Should approx. be in decreasing value
        self.assertEqual(scores.shape, (krc.nb_classes - 1,))

    def test_clever_l2_same_target(self):
        batch_size = 100
        (x_train, y_train), (x_test, _), _, _ = load_mnist()

        # Get the classifier
        krc = self._create_krclassifier()
        krc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=2, verbose=0)

        scores = clever(
            krc,
            x_test[0],
            5,
            5,
            3,
            2,
            target=np.argmax(krc.predict(x_test[:1])),
            c_init=1,
            pool_factor=10,
            verbose=False,
        )
        self.assertIsNone(scores[0], msg="Clever scores for the predicted class should be `None`.")

    def test_1_wasserstein_distance(self):
        nb_train = 1000
        nb_test = 100
        batch_size = 3
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()

        x_train = x_train[0:nb_train]
        x_test = x_test[0:nb_test]
        weights = np.ones_like(x_train)

        wd_0 = wasserstein_distance(x_train[:batch_size], x_train[:batch_size])
        wd_1 = wasserstein_distance(x_train[:batch_size], x_test[:batch_size])
        wd_2 = wasserstein_distance(
            x_train[:batch_size], x_train[:batch_size], weights[:batch_size], weights[:batch_size]
        )

        np.testing.assert_array_equal(wd_0, np.asarray([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(wd_1, np.asarray([0.04564, 0.01235, 0.04787]), decimal=4)

        np.testing.assert_array_equal(x_train.shape, np.asarray([nb_train, 28, 28, 1]))
        np.testing.assert_array_equal(x_test.shape, np.asarray([nb_test, 28, 28, 1]))

        np.testing.assert_array_equal(wd_2, np.asarray([0.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
