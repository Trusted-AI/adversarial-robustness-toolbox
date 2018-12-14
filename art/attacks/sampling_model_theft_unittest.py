from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from art.attacks import SamplingModelTheft
from art.classifiers import KerasClassifier
from art.utils import load_dataset, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 1000, 10


class TestSamplingModelTheft(unittest.TestCase):
    """
    A unittest class for testing SamplingModelTheft attack.
    """
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_krclassifier(self):
        """
        First test with the KerasClassifier.
        :return:
        """
        (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        m0 = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')])
        m0.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        k0 = KerasClassifier((min_, max_), model=m0)
        k0.fit(x_train, y_train, nb_epochs=2, batch_size=128)

        m1 = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')])
        m1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        k1 = KerasClassifier((min_, max_), model=m1)

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=False)
        datagen.fit(x_train)
        fit_datagen = lambda x, y: datagen.flow(x, y)
        att = SamplingModelTheft(x_test, fit_datagen=fit_datagen)
        k1 = att.steal(k0, k1, 1000, nb_epochs=2)

        y0 = k0.predict(x_train)
        y1 = k1.predict(x_train)

        agree = np.sum(y0.argmax(axis=1) == y1.argmax(axis=1)) / len(x_train)
        self.assertTrue(agree >= 0.1)


if __name__ == '__main__':
    unittest.main()
