from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from art.classifiers import KerasClassifier
from art.attacks.sampling_model_theft import SamplingModelTheft
from art.defences import ReverseSigmoid
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from art.utils import load_dataset
import numpy as np


class TestSamplingModelTheft(unittest.TestCase):
    """
    A unittest class for testing SamplingModelTheft attack.
    """
    def test_krclassifier(self):
        """
        First test with the KerasClassifier.
        :return:
        """
        (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))
        
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
        k0.fit(x_train, y_train, nb_epochs=5, batch_size=128)
        
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
        att = SamplingModelTheft(x_test)
        k1 = att.steal(k0,k1,10000, epochs=5)
        
        y0 = k0.predict(x_train)
        y1 = k1.predict(x_train)
        
        agree = np.sum(y0.argmax(axis=1)==y1.argmax(axis=1)) / len(x_train)
        
        self.assertTrue(agree>=0.9)



if __name__ == '__main__':
    unittest.main()
