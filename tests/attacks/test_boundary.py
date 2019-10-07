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
import tensorflow as tf

from art.attacks import BoundaryAttack
from art.classifiers import KerasClassifier
from art.utils import load_dataset, random_targets, master_seed
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from art.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger('testLogger')

NB_TRAIN = 100
NB_TEST = 20


class TestBoundary(unittest.TestCase):
    """
    A unittest class for testing the Boundary attack.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

    def setUp(self):
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # First targeted attack
        boundary = BoundaryAttack(classifier=tfc, targeted=True, max_iter=200, delta=0.5)
        params = {'y': random_targets(self.y_test, tfc.nb_classes())}
        x_test_adv = boundary.generate(self.x_test, **params)
        # expected_x_test_adv_1 = np.asarray([0.42622495, 0.0, 0.0, 0.33005068, 0.2277837, 0.0,
        #                                     0.18348512, 0.42622495, 0.27452883, 0.0, 0.0, 0.0,
        #                                     0.1653487, 0.70523715, 0.7367977, 0.7974912, 0.28579983, 0.0,
        #                                     0.36499417, 0.0, 0.0, 0.0, 0.42622495, 0.0,
        #                                     0.26680174, 0.42622495, 0.0, 0.19260764])
        # expected_x_test_adv_2 = np.asarray([0.0459, 0., 0., 0.0756, 0.2048, 0.037, 0., 0.,
        #                                     0.0126, 0.4338, 0.1566, 0.3061, 0., 0.296, 0.8318, 0.7267,
        #                                     0.2252, 0.074, 0., 0.1208, 0.4362, 0., 0., 0.,
        #                                     0., 0.0359, 0., 0.1191])
        #
        # expected_x_test_adv_3 = np.asarray([0.0671, 0.0644, 0.3012, 0., 0., 0., 0.3407, 0.,
        #                                     0.1507, 0.0478, 0.3253, 0., 0.3334, 0.3473, 1., 0.8649,
        #                                     0.5639, 0.5198, 0., 0., 0.6173, 0., 0.3116, 0.,
        #                                     0.3937, 0.6173, 0., 0.0021])
        # try:
        #     np.testing.assert_array_almost_equal(x_test_adv[2, 14, :, 0], expected_x_test_adv_1, decimal=4)
        # except AssertionError:
        #     try:
        #         np.testing.assert_array_almost_equal(x_test_adv[2, 14, :, 0], expected_x_test_adv_2, decimal=4)
        #     except AssertionError:
        #         np.testing.assert_array_almost_equal(x_test_adv[2, 14, :, 0], expected_x_test_adv_3, decimal=4)
        self.assertLessEqual(np.max(x_test_adv), 1.0)
        self.assertGreaterEqual(np.min(x_test_adv), 0.0)

        y_pred_adv = tfc.predict(x_test_adv)
        y_pred_adv_expected = np.asarray([1.57103419e-01, -7.31061280e-01, -4.03979905e-02, -4.79048371e-01,
                                          9.37852338e-02, -8.01057637e-01, -4.77534801e-01, 1.08687377e+00,
                                          -3.06577891e-01, -5.74976981e-01])
        np.testing.assert_array_almost_equal(y_pred_adv[0], y_pred_adv_expected, decimal=4)

        # Second untargeted attack
        boundary = BoundaryAttack(classifier=tfc, targeted=False, max_iter=20)
        x_test_adv = boundary.generate(self.x_test)

        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(tfc.predict(self.x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Clean-up session
        sess.close()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      'v2 as backend.')
    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc = get_classifier_kr()

        # First targeted attack
        boundary = BoundaryAttack(classifier=krc, targeted=True, max_iter=20)
        params = {'y': random_targets(self.y_test, krc.nb_classes())}
        x_test_adv = boundary.generate(self.x_test, **params)

        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second untargeted attack
        boundary = BoundaryAttack(classifier=krc, targeted=False, max_iter=20)
        x_test_adv = boundary.generate(self.x_test)

        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(krc.predict(self.x_test), axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Clean-up session
        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        x_test = np.swapaxes(self.x_test, 1, 3).astype(np.float32)

        # First targeted attack
        boundary = BoundaryAttack(classifier=ptc, targeted=True, max_iter=20)
        params = {'y': random_targets(self.y_test, ptc.nb_classes())}
        x_test_adv = boundary.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second untargeted attack
        boundary = BoundaryAttack(classifier=ptc, targeted=False, max_iter=20)
        x_test_adv = boundary.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(ptc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = BoundaryAttack(classifier=classifier)

        self.assertIn('For `BoundaryAttack` classifier must be an instance of `art.classifiers.classifier.Classifier`, '
                      'the provided classifier is instance of (<class \'object\'>,).', str(context.exception))


class TestBoundaryVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

    def setUp(self):
        master_seed(1234)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_clipped(self):
        classifier, _ = get_iris_classifier_kr()
        attack = BoundaryAttack(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with boundary adversarial examples: %.2f%%', (accuracy * 100))

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_unbounded(self):
        classifier, _ = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = BoundaryAttack(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with boundary adversarial examples: %.2f%%', (accuracy * 100))

    def test_iris_tf(self):
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = BoundaryAttack(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with boundary adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(self.y_test, nb_classes=3)
        attack = BoundaryAttack(classifier, targeted=True, max_iter=10)
        x_test_adv = attack.generate(self.x_test, **{'y': targets})
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        accuracy = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
        logger.info('Success rate of targeted boundary on Iris: %.2f%%', (accuracy * 100))

    def test_iris_pt(self):
        classifier = get_iris_classifier_pt()
        attack = BoundaryAttack(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test.astype(np.float32))
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with boundary adversarial examples: %.2f%%', (accuracy * 100))

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

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test, y=self.y_test)

            attack = BoundaryAttack(classifier, targeted=False, delta=0.01, epsilon=0.01, step_adapt=0.667, max_iter=50,
                                    num_trial=25, sample_size=20, init_size=100)
            x_test_adv = attack.generate(self.x_test)
            self.assertFalse((self.x_test == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test, axis=1) == preds_adv).all())
            accuracy = np.sum(preds_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with BoundaryAttack adversarial '
                                                                         'examples: %.2f%%', (accuracy * 100))


if __name__ == '__main__':
    unittest.main()
