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

import keras.backend as k
import numpy as np

from art.attacks import HopSkipJump
from art.classifiers import KerasClassifier
from art.utils import load_dataset, random_targets, master_seed
from tests.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from tests.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)

NB_TRAIN = 100
NB_TEST = 10


class TestHopSkipJump(unittest.TestCase):
    """
    A unittest class for testing the HopSkipJump attack.
    """

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

    def test_tensorflow_mnist(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # First targeted attack and norm=2
        hsj = HopSkipJump(classifier=tfc, targeted=True, max_iter=2, max_eval=100, init_eval=10)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = hsj.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # First targeted attack and norm=np.inf
        hsj = HopSkipJump(classifier=tfc, targeted=True, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = hsj.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second untargeted attack and norm=2
        hsj = HopSkipJump(classifier=tfc, targeted=False, max_iter=2, max_eval=100, init_eval=10)
        x_test_adv = hsj.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(tfc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Second untargeted attack and norm=np.inf
        hsj = HopSkipJump(classifier=tfc, targeted=False, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        x_test_adv = hsj.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(tfc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # Clean-up session
        sess.close()

    def test_keras_mnist(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build KerasClassifier
        krc = get_classifier_kr()

        # First targeted attack and norm=2
        hsj = HopSkipJump(classifier=krc, targeted=True, max_iter=2, max_eval=100, init_eval=10)
        params = {'y': random_targets(y_test, krc.nb_classes())}
        x_test_adv = hsj.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # First targeted attack and norm=np.inf
        hsj = HopSkipJump(classifier=krc, targeted=True, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        params = {'y': random_targets(y_test, krc.nb_classes())}
        x_test_adv = hsj.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second untargeted attack and norm=2
        hsj = HopSkipJump(classifier=krc, targeted=False, max_iter=2, max_eval=100, init_eval=10)
        x_test_adv = hsj.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(krc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Second untargeted attack and norm=np.inf
        hsj = HopSkipJump(classifier=krc, targeted=False, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        x_test_adv = hsj.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(krc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # Clean-up session
        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
        x_test_original = x_test.copy()

        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # First targeted attack and norm=2
        hsj = HopSkipJump(classifier=ptc, targeted=True, max_iter=2, max_eval=100, init_eval=10)
        params = {'y': random_targets(y_test, ptc.nb_classes())}
        x_test_adv = hsj.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # First targeted attack and norm=np.inf
        hsj = HopSkipJump(classifier=ptc, targeted=True, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        params = {'y': random_targets(y_test, ptc.nb_classes())}
        x_test_adv = hsj.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second untargeted attack and norm=2
        hsj = HopSkipJump(classifier=ptc, targeted=False, max_iter=2, max_eval=100, init_eval=10)
        x_test_adv = hsj.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(ptc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Second untargeted attack and norm=np.inf
        hsj = HopSkipJump(classifier=ptc, targeted=False, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        x_test_adv = hsj.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(ptc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = HopSkipJump(classifier=classifier)

        self.assertIn('For `HopSkipJump` classifier must be an instance of `art.classifiers.classifier.Classifier`, the'
                      ' provided classifier is instance of (<class \'object\'>,).', str(context.exception))

    def test_pytorch_resume(self):
        (_, _), (x_test, y_test) = self.mnist
        x_test = np.reshape(x_test, (x_test.shape[0], 1, 28, 28)).astype(np.float32)

        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # HSJ attack
        hsj = HopSkipJump(classifier=ptc, targeted=True, max_iter=10, max_eval=100, init_eval=10)

        params = {'y': y_test[2:3], 'x_adv_init': x_test[2:3]}
        x_test_adv1 = hsj.generate(x_test[0:1], **params)
        diff1 = np.linalg.norm(x_test_adv1 - x_test)

        params.update(resume=True, x_adv_init=x_test_adv1)
        x_test_adv2 = hsj.generate(x_test[0:1], **params)
        params.update(x_adv_init=x_test_adv2)
        x_test_adv2 = hsj.generate(x_test[0:1], **params)
        diff2 = np.linalg.norm(x_test_adv2 - x_test)

        self.assertGreater(diff1, diff2)

    def test_keras_iris_clipped(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

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

        # Clean-up session
        k.clear_session()

    def test_keras_iris_unbounded(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)

        # Norm=2
        attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', (acc * 100))

        # Norm=np.inf
        attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', (acc * 100))

        # Clean-up session
        k.clear_session()

    def test_tensorflow_iris(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, sess = get_iris_classifier_tf()

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

        # Clean-up session
        sess.close()

    def test_pytorch_iris(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()
        x_test = x_test.astype(np.float32)

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

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier, ScikitlearnExtraTreeClassifier
        from art.classifiers.scikitlearn import ScikitlearnAdaBoostClassifier, ScikitlearnBaggingClassifier
        from art.classifiers.scikitlearn import ScikitlearnExtraTreesClassifier, ScikitlearnGradientBoostingClassifier
        from art.classifiers.scikitlearn import ScikitlearnRandomForestClassifier, ScikitlearnLogisticRegression
        from art.classifiers.scikitlearn import ScikitlearnSVC

        scikitlearn_test_cases = {DecisionTreeClassifier: ScikitlearnDecisionTreeClassifier,
                                  ExtraTreeClassifier: ScikitlearnExtraTreeClassifier,
                                  AdaBoostClassifier: ScikitlearnAdaBoostClassifier,
                                  BaggingClassifier: ScikitlearnBaggingClassifier,
                                  ExtraTreesClassifier: ScikitlearnExtraTreesClassifier,
                                  GradientBoostingClassifier: ScikitlearnGradientBoostingClassifier,
                                  RandomForestClassifier: ScikitlearnRandomForestClassifier,
                                  LogisticRegression: ScikitlearnLogisticRegression,
                                  SVC: ScikitlearnSVC,
                                  LinearSVC: ScikitlearnSVC}

        (_, _), (x_test, y_test) = self.iris
        x_test_original = x_test.copy()

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=x_test, y=y_test)

            # Norm=2
            attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10)
            x_test_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with HopSkipJump adversarial '
                        'examples: %.2f%%', (acc * 100))

            # Norm=np.inf
            attack = HopSkipJump(classifier, targeted=False, max_iter=2, max_eval=100, init_eval=10, norm=np.Inf)
            x_test_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with HopSkipJump adversarial '
                        'examples: %.2f%%', (acc * 100))

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)


if __name__ == '__main__':
    unittest.main()
