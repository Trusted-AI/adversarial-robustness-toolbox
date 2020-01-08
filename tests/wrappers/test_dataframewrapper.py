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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import pandas as pd
import numpy as np

from art.wrappers import DataframeWrapper
from art.utils import load_dataset, random_targets, master_seed
from tests.utils_test import get_iris_classifier_kr, get_iris_classifier_pt

from art.attacks import HopSkipJump

logger = logging.getLogger('testLogger')


class TestDataframeWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #  Get Default DataFrame
        cls.test_data = pd.DataFrame({'Name': [0, 1, 2, 3], 'Age': [20, 21, 19, 18]})

        #  Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        print(type(x_test))
        cls.iris = (DataframeWrapper(pd.DataFrame(x_train)), y_train), (DataframeWrapper(pd.DataFrame(x_test)), y_test)

    def setUp(self):
        master_seed(1234)

    def test_class_creation(self):
        #  Create from new
        test_frame = DataframeWrapper(self.test_data)
        self.assertTrue(isinstance(test_frame, DataframeWrapper))
        self.assertTrue(isinstance(test_frame.dataframe, pd.DataFrame))
        self.assertTrue((test_frame == test_frame.dataframe.to_numpy()).all())

        #  Create from template
        test_frame2 = test_frame
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertTrue(isinstance(test_frame2.dataframe, pd.DataFrame))
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())
        self.assertTrue((test_frame == test_frame2).all())
        self.assertTrue((test_frame.dataframe.to_numpy() == test_frame2.dataframe.to_numpy()).all())

        #  Create from template slice
        test_frame2 = test_frame[0]
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertTrue(isinstance(test_frame2.dataframe, pd.DataFrame))
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())

        test_frame2 = test_frame[:2]
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertTrue(isinstance(test_frame2.dataframe, pd.DataFrame))
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())

        test_frame2 = test_frame[:2]
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertTrue(isinstance(test_frame2.dataframe, pd.DataFrame))
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())

    def test_copy(self):
        #  Test copy during creation
        test_frame = DataframeWrapper(self.test_data, copy=False)
        self.assertTrue(test_frame.dataframe is self.test_data)

        test_frame = DataframeWrapper(self.test_data)
        self.assertFalse(test_frame.dataframe is self.test_data)

        #  Test explicit copy
        test_frame2 = test_frame.copy()
        self.assertFalse(test_frame2 is test_frame)
        self.assertFalse(test_frame2.dataframe is test_frame.dataframe)

        #  Test explicit copy slice
        test_frame2 = test_frame[0].copy()
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())
        self.assertFalse(test_frame2 is test_frame[0])

        test_frame2 = test_frame[:2].copy()
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())
        self.assertFalse(test_frame2 is test_frame[:2])

        test_frame2 = test_frame[2:].copy()
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())
        self.assertFalse(test_frame2 is test_frame[2:])

    def test_operators(self):
        test_frame = DataframeWrapper(self.test_data)

        #  Test the basic operators and broadcasting
        test_frame = test_frame + 1
        self.assertTrue((test_frame == test_frame.dataframe.to_numpy()).all())
        test_frame = test_frame - 1
        self.assertTrue((test_frame == test_frame.dataframe.to_numpy()).all())
        test_frame = test_frame * 2
        self.assertTrue((test_frame == test_frame.dataframe.to_numpy()).all())
        test_frame = test_frame / 2
        self.assertTrue((test_frame == test_frame.dataframe.to_numpy()).all())

    def test_basic_np_functions(self):
        test_frame = DataframeWrapper(self.test_data)

        #  Test the common numpy functions

        #  These first two should fail because they don't actually care what the input class type is.
        test_frame2 = np.concatenate((test_frame, test_frame))
        with self.assertRaises(AttributeError):
            print(test_frame2.dataframe)
        test_frame2 = np.stack((test_frame, test_frame))
        with self.assertRaises(AttributeError):
            print(test_frame2.dataframe)

        #  Now we check some of the commonly used numpy or ndarray functions
        test_frame2 = np.repeat(test_frame, 2, axis=0)
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertTrue(np.shape(test_frame2) == np.shape(test_frame2.dataframe.to_numpy()))
        self.assertTrue((test_frame2 == test_frame2.dataframe.to_numpy()).all())

        test_frame2 = np.repeat(test_frame, 2, axis=1)
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertFalse(np.shape(test_frame2) == np.shape(test_frame2.dataframe.to_numpy()))

        test_frame2 = np.reshape(test_frame, (8, 1))
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertFalse(np.shape(test_frame2) == np.shape(test_frame2.dataframe.to_numpy()))
        test_frame2 = test_frame.reshape((8, 1))
        self.assertTrue(isinstance(test_frame2, DataframeWrapper))
        self.assertFalse(np.shape(test_frame2) == np.shape(test_frame2.dataframe.to_numpy()))

        #  argmax will return a Dataframewrapper object also with a modified dataframe, technically this is a bug
        test_max = np.max(test_frame, axis=0)
        self.assertTrue(isinstance(test_max, DataframeWrapper))
        self.assertTrue((test_max == np.max(self.test_data.to_numpy(), axis=0)).all())

        test_max = np.max(test_frame, axis=1)
        self.assertTrue(isinstance(test_max, DataframeWrapper))
        self.assertFalse(np.shape(test_max) == np.shape(test_max.dataframe.to_numpy()))
        self.assertTrue((test_max == np.max(self.test_data.to_numpy(), axis=1)).all())

        test_frame.fill(2)
        self.assertTrue(isinstance(test_frame, DataframeWrapper))
        self.assertTrue(np.shape(test_frame) == np.shape(test_frame.dataframe.to_numpy()))

    def test_iris_tf(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

        # Test untargeted attack and norm=2
        attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', (acc * 100))

        # Test untargeted attack and norm=np.inf
        attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', (acc * 100))

        # Test targeted attack and norm=2
        targets = random_targets(y_test, nb_classes=3)
        attack = HopSkipJump(classifier, targeted=True, max_iter=2, max_eval=100, init_eval=10)
        x_test_adv = attack.generate(x_test, **{'y': targets})
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
        logger.info('Success rate of targeted HopSkipJump on Iris: %.2f%%', (acc * 100))

        # Test targeted attack and norm=np.inf
        targets = random_targets(y_test, nb_classes=3)
        attack = HopSkipJump(classifier, targeted=True, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        x_test_adv = attack.generate(x_test, **{'y': targets})
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
        logger.info('Success rate of targeted HopSkipJump on Iris: %.2f%%', (acc * 100))

    @unittest.skip("demonstrating skipping")
    def test_iris_pt(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()

        # Norm=2
        attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', (acc * 100))

        # Norm=np.inf
        attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', (acc * 100))

if __name__ == '__main__':
    unittest.main()
