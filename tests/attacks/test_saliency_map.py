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

import numpy as np

from art.attacks import SaliencyMapMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset, get_labels_np_array, to_categorical, master_seed
from tests.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from tests.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 2


class TestSaliencyMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(1234)

    def test_keras_mnist(self):

        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Keras classifier
        classifier = get_classifier_kr()

        scores = classifier._model.evaluate(x_train, y_train)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))

        scores = classifier._model.evaluate(x_test, y_test)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # targeted

        # Generate random target classes
        nb_classes = np.unique(np.argmax(y_test, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=NB_TEST)
        while (targets == np.argmax(y_test, axis=1)).any():
            targets = np.random.randint(nb_classes, size=NB_TEST)

        # Perform attack
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(x_test, y=to_categorical(targets, nb_classes))

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # untargeted
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_tensorflow_mnist(self):

        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Create basic CNN on MNIST using TensorFlow
        classifier, sess = get_classifier_tf()

        scores = get_labels_np_array(classifier.predict(x_train))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(x_test))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

        # targeted
        # Generate random target classes
        nb_classes = np.unique(np.argmax(y_test, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=NB_TEST)
        while (targets == np.argmax(y_test, axis=1)).any():
            targets = np.random.randint(nb_classes, size=NB_TEST)

        # Perform attack
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(x_test, y=to_categorical(targets, nb_classes))

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # untargeted
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_pytorch_mnist(self):

        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
        x_test_original = x_test.copy()

        # Create basic PyTorch model
        classifier = get_classifier_pt()

        scores = get_labels_np_array(classifier.predict(x_train))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(x_test))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('\n[PyTorch, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

        # targeted
        # Generate random target classes
        nb_classes = np.unique(np.argmax(y_test, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=NB_TEST)
        while (targets == np.argmax(y_test, axis=1)).any():
            targets = np.random.randint(nb_classes, size=NB_TEST)

        # Perform attack
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(x_test, y=to_categorical(targets, nb_classes))

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # untargeted
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = SaliencyMapMethod(classifier=classifier)

        self.assertIn('For `SaliencyMapMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = SaliencyMapMethod(classifier=classifier)

        self.assertIn('For `SaliencyMapMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    def test_keras_iris_vector_clipped(self):

        (_, _), (x_test, y_test) = self.iris

        classifier = get_iris_classifier_kr()

        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', (accuracy * 100))

    def test_keras_iris_vector_unbounded(self):

        (_, _), (x_test, y_test) = self.iris

        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

    def test_tensorflow_iris_vector(self):

        (_, _), (x_test, y_test) = self.iris

        classifier, _ = get_iris_classifier_tf()

        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', (accuracy * 100))

    def test_pytorch_iris_vector(self):

        (_, _), (x_test, y_test) = self.iris

        classifier = get_iris_classifier_pt()

        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', (accuracy * 100))

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression

        from art.classifiers.scikitlearn import ScikitlearnLogisticRegression

        scikitlearn_test_cases = {LogisticRegression: ScikitlearnLogisticRegression}  # ,
        # SVC: ScikitlearnSVC,
        # LinearSVC: ScikitlearnSVC}

        (_, _), (x_test, y_test) = self.iris
        x_test_original = x_test.copy()

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=x_test, y=y_test)

            attack = SaliencyMapMethod(classifier, theta=1, batch_size=128)
            x_test_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
            accuracy = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with JSMA adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)


if __name__ == '__main__':
    unittest.main()
