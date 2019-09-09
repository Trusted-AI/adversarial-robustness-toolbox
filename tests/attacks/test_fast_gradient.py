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

import tensorflow as tf
import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.defences import FeatureSqueezing
from art.utils import load_dataset, get_labels_np_array, master_seed, random_targets
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from art.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11


@unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                  ' v2 as backend.')
class TestFastGradientMethodImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

        # Keras classifier
        cls.classifier_k = get_classifier_kr()

        scores = cls.classifier_k._model.evaluate(x_train, y_train)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))
        scores = cls.classifier_k._model.evaluate(x_test, y_test)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # Create basic CNN on MNIST using TensorFlow
        cls.classifier_tf, sess = get_classifier_tf()

        scores = get_labels_np_array(cls.classifier_tf.predict(x_train))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(cls.classifier_tf.predict(x_test))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

        # Create basic PyTorch model
        cls.classifier_py = get_classifier_pt()
        x_train, x_test = np.swapaxes(x_train, 1, 3), np.swapaxes(x_test, 1, 3)

        scores = get_labels_np_array(cls.classifier_py.predict(x_train.astype(np.float32)))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(cls.classifier_py.predict(x_test.astype(np.float32)))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

    def setUp(self):
        master_seed(1234)

    def test_mnist(self):
        # Define all backends to test
        backends = {'keras': self.classifier_k,
                    'tf': self.classifier_tf,
                    'pytorch': self.classifier_py}

        for classifier_name, classifier in backends.items():
            if classifier_name == 'pytorch':
                self._swap_axes()
                self.x_train = self.x_train.astype(np.float32)
                self.x_test = self.x_test.astype(np.float32)

            self._test_backend_mnist(classifier)

            if classifier_name == 'pytorch':
                self._swap_axes()
                self.x_train = self.x_train.astype(np.float64)
                self.x_test = self.x_test.astype(np.float64)

    def _swap_axes(self):
        self.x_train = np.swapaxes(self.x_train, 1, 3)
        self.x_test = np.swapaxes(self.x_test, 1, 3)

    def _test_backend_mnist(self, classifier):
        # Test FGSM with np.inf norm
        attack = FastGradientMethod(classifier, eps=1, batch_size=2)
        x_test_adv = attack.generate(self.x_test)
        x_train_adv = attack.generate(self.x_train)

        self.assertFalse((self.x_train == x_train_adv).all())
        self.assertFalse((self.x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((self.y_train == train_y_pred).all())
        self.assertFalse((self.y_test == test_y_pred).all())

        accuracy = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(self.y_train, axis=1)) / self.y_train.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial train examples: %.2f%%', (accuracy * 100))

        accuracy = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial test examples: %.2f%%', (accuracy * 100))

        # Test minimal perturbations
        attack_params = {"minimal": True, "eps_step": 0.1, "eps": 1.0}
        attack.set_params(**attack_params)

        x_train_adv_min = attack.generate(self.x_train)
        x_test_adv_min = attack.generate(self.x_test)

        self.assertFalse((x_train_adv_min == x_train_adv).all())
        self.assertFalse((x_test_adv_min == x_test_adv).all())

        self.assertFalse((self.x_train == x_train_adv_min).all())
        self.assertFalse((self.x_test == x_test_adv_min).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv_min))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv_min))

        self.assertFalse((self.y_train == train_y_pred).all())
        self.assertFalse((self.y_test == test_y_pred).all())

        accuracy = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(self.y_train, axis=1)) / self.y_train.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial train examples with minimal perturbation: %.2f%%',
                    (accuracy * 100))

        accuracy = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial test examples with minimal perturbation: %.2f%%',
                    (accuracy * 100))

        # L_1 norm
        attack = FastGradientMethod(classifier, eps=1, norm=1, batch_size=128)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test == test_y_pred).all())
        accuracy = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial test examples with L1 norm: %.2f%%', (accuracy * 100))

        # L_2 norm
        attack = FastGradientMethod(classifier, eps=1, norm=2, batch_size=128)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test == test_y_pred).all())
        accuracy = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial test examples with L2 norm: %.2f%%', (accuracy * 100))

        # Test random initialisations
        attack = FastGradientMethod(classifier, num_random_init=3)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test == test_y_pred).all())
        accuracy = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial test examples with 3 random initialisations: %.2f%%',
                    (accuracy * 100))

    def test_with_defences(self):
        # Get the ready-trained Keras model
        model = self.classifier_k._model
        fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
        classifier = KerasClassifier(model=model, clip_values=(0, 1), defences=fs)

        attack = FastGradientMethod(classifier, eps=1, batch_size=128)
        x_train_adv = attack.generate(self.x_train)
        x_test_adv = attack.generate(self.x_test)

        self.assertFalse((self.x_train == x_train_adv).all())
        self.assertFalse((self.x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((self.y_train == train_y_pred).all())
        self.assertFalse((self.y_test == test_y_pred).all())

        predictions = classifier.predict(x_train_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.y_train, axis=1)) / self.y_train.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial train examples with feature squeezing: %.2f%%',
                    (accuracy * 100))

        predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on MNIST with FGM adversarial test examples: %.2f%%', (accuracy * 100))

    def _test_mnist_targeted(self, classifier):
        # Test FGSM with np.inf norm
        attack = FastGradientMethod(classifier, eps=1.0, targeted=True)

        pred_sort = classifier.predict(self.x_test).argsort(axis=1)
        y_test_adv = np.zeros((self.x_test.shape[0], 10))
        for i in range(self.x_test.shape[0]):
            y_test_adv[i, pred_sort[i, -2]] = 1.0

        attack_params = {"minimal": True, "eps_step": 0.01, "eps": 1.0}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(self.x_test, y=y_test_adv)
        self.assertFalse((self.x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertEqual(y_test_adv.shape, test_y_pred.shape)
        self.assertGreaterEqual((y_test_adv == test_y_pred).sum(), self.x_test.shape[0] // 2)

    def test_mnist_targeted(self):
        # Define all backends to test
        backends = {'keras': self.classifier_k,
                    'tf': self.classifier_tf,
                    'pytorch': self.classifier_py}

        for classifier_name, classifier in backends.items():
            if classifier_name == 'pytorch':
                self._swap_axes()
                self.x_train = self.x_train.astype(np.float32)
                self.x_test = self.x_test.astype(np.float32)

            self._test_mnist_targeted(classifier)

            if classifier_name == 'pytorch':
                self._swap_axes()
                self.x_train = self.x_train.astype(np.float64)
                self.x_test = self.x_test.astype(np.float64)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = FastGradientMethod(classifier=classifier)

        self.assertIn('For `FastGradientMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = FastGradientMethod(classifier=classifier)

        self.assertIn('For `FastGradientMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))


class TestFastGradientVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

    def setUp(self):
        master_seed(1234)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                      ' v2 as backend.')
    def test_iris_k_clipped(self):
        classifier, _ = get_iris_classifier_kr()

        # Test untargeted attack
        attack = FastGradientMethod(classifier, eps=.1)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with FGM adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(self.y_test, nb_classes=3)
        attack = FastGradientMethod(classifier, targeted=True, eps=.1)
        x_test_adv = attack.generate(self.x_test, **{'y': targets})
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
        logger.info('Success rate of targeted FGM on Iris: %.2f%%', (accuracy * 100))

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                      ' v2 as backend.')
    def test_iris_k_unbounded(self):
        classifier, _ = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = FastGradientMethod(classifier, eps=1)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv > 1).any())
        self.assertTrue((x_test_adv < 0).any())

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with FGM adversarial examples: %.2f%%', (accuracy * 100))

    def test_iris_tf(self):
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = FastGradientMethod(classifier, eps=.1)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with FGM adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(self.y_test, nb_classes=3)
        attack = FastGradientMethod(classifier, targeted=True, eps=.1, batch_size=128)
        x_test_adv = attack.generate(self.x_test, **{'y': targets})
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
        logger.info('Success rate of targeted FGM on Iris: %.2f%%', (accuracy * 100))

    def test_iris_pt(self):
        classifier = get_iris_classifier_pt()

        # Test untargeted attack
        attack = FastGradientMethod(classifier, eps=.1)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with FGM adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(self.y_test, nb_classes=3)
        attack = FastGradientMethod(classifier, targeted=True, eps=.1, batch_size=128)
        x_test_adv = attack.generate(self.x_test, **{'y': targets})
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
        logger.info('Success rate of targeted FGM on Iris: %.2f%%', (accuracy * 100))

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.classifiers.scikitlearn import ScikitlearnLogisticRegression, ScikitlearnSVC

        scikitlearn_test_cases = {LogisticRegression: ScikitlearnLogisticRegression,
                                  SVC: ScikitlearnSVC,
                                  LinearSVC: ScikitlearnSVC}

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test, y=self.y_test)

            # Test untargeted attack
            eps = 0.1
            attack = FastGradientMethod(classifier, eps=eps)
            x_test_adv = attack.generate(self.x_test)
            np.testing.assert_array_almost_equal(np.abs(x_test_adv - self.x_test), eps, decimal=5)
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
            accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with FGM adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Test targeted attack
            targets = random_targets(self.y_test, nb_classes=3)
            attack = FastGradientMethod(classifier, targeted=True, eps=0.1, batch_size=128)
            x_test_adv = attack.generate(self.x_test, **{'y': targets})
            self.assertFalse((self.x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
            logger.info('Success rate of ' + classifier.__class__.__name__ + ' on targeted FGM on Iris: %.2f%%',
                        (accuracy * 100))


if __name__ == '__main__':
    unittest.main()
