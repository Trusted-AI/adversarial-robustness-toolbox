from __future__ import absolute_import, division, print_function

import tensorflow as tf
import unittest
import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from src.attacks.newtonfool import NewtonFool
from src.classifiers.tensorflow import TFClassifier
from src.classifiers.keras import KerasClassifier
from src.utils import load_mnist


class TestNewtonFool(unittest.TestCase):
    """
    A unittest class for testing the NewtonFool attack.
    """
    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build a TFClassifier
        # Define input and output placeholders
        self._input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self._output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(self._input_ph, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        self._logits = tf.layers.dense(fc, 10)

        # Train operator
        self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            logits=self._logits, onehot_labels=self._output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self._train = optimizer.minimize(self._loss)

        # Tensorflow session and initialization
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 20
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Train the classifier
        tfc = TFClassifier(None, self._input_ph, self._logits, False,
                           self._output_ph, self._train, self._loss,
                           None, self._sess)
        tfc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=1)

        # Attack
        nf = NewtonFool(tfc)
        nf.set_params(max_iter=2)
        x_test_adv = nf.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        y_pred = tfc.predict(x_test)
        y_pred_adv = tfc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 20
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        k.set_learning_phase(1)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

        # Get classifier
        krc = KerasClassifier((0, 1), model, use_logits=False)

        # Attack
        k.set_learning_phase(0)
        nf = NewtonFool(krc)
        nf.set_params(max_iter=2)
        x_test_adv = nf.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        y_pred = krc.predict(x_test)
        y_pred_adv = krc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())


if __name__ == '__main__':
    unittest.main()



