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

# import keras
# import keras.backend as k
import numpy as np

from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod

# from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import random_targets

# from art.utils import to_categorical

from tests.utils import TestBase, master_seed
from tests.utils import get_image_classifier_tf

# from test.utils import get_image_classifier_kr, get_image_classifier_pt
# from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestCarlini(TestBase):
    """
    A unittest class for testing the Carlini L2 attack.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 10
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def setUp(self):
        master_seed(seed=1234)
        super().setUp()

    def test_tensorflow_failure_attack_L2(self):
        """
        Test the corner case when attack is failed.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf(from_logits=True)

        # Failure attack
        cl2m = CarliniL2Method(
            classifier=tfc,
            targeted=True,
            max_iter=0,
            binary_search_steps=0,
            learning_rate=0,
            initial_const=1,
            verbose=False,
        )
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = cl2m.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        np.testing.assert_array_almost_equal(self.x_test_mnist, x_test_adv, decimal=3)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_tensorflow_mnist_L2(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf(from_logits=True)

        # First attack
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=10, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = cl2m.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("CW2 Target: %s", target)
        logger.debug("CW2 Actual: %s", y_pred_adv)
        logger.info("CW2 Success Rate: %.2f", (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack, no batching
        cl2m = CarliniL2Method(classifier=tfc, targeted=False, max_iter=10, batch_size=1, verbose=False)
        x_test_adv = cl2m.generate(self.x_test_mnist)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("CW2 Target: %s", target)
        logger.debug("CW2 Actual: %s", y_pred_adv)
        logger.info("CW2 Success Rate: %.2f", (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        # Clean-up session
        if sess is not None:
            sess.close()

    # @unittest.skipIf(
    #     not (int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3),
    #     reason="Minimal version of Keras or TensorFlow required.",
    # )
    # def test_keras_mnist_L2(self):
    #     """
    #     Second test with the KerasClassifier.
    #     :return:
    #     """
    #     x_test_original = self.x_test_mnist.copy()
    #
    #     # Build KerasClassifier
    #     krc = get_image_classifier_kr(from_logits=True)
    #
    #     # First attack
    #     cl2m = CarliniL2Method(classifier=krc, targeted=True, max_iter=10, verbose=False)
    #     y_target = [6, 6, 7, 4, 9, 7, 9, 0, 1, 0]
    #     x_test_adv = cl2m.generate(self.x_test_mnist, y=to_categorical(y_target, nb_classes=10))
    #     self.assertFalse((self.x_test_mnist == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #     y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
    #     logger.debug("CW2 Target: %s", y_target)
    #     logger.debug("CW2 Actual: %s", y_pred_adv)
    #     logger.info("CW2 Success Rate: %.2f", (np.sum(y_target == y_pred_adv) / float(len(y_target))))
    #     self.assertTrue((y_target == y_pred_adv).any())
    #
    #     # Second attack
    #     cl2m = CarliniL2Method(classifier=krc, targeted=False, max_iter=10, verbose=False)
    #     x_test_adv = cl2m.generate(self.x_test_mnist)
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #     y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
    #     logger.debug("CW2 Target: %s", y_target)
    #     logger.debug("CW2 Actual: %s", y_pred_adv)
    #     logger.info("CW2 Success Rate: %.2f", (np.sum(y_target != y_pred_adv) / float(len(y_target))))
    #     self.assertTrue((y_target != y_pred_adv).any())
    #
    #     # Check that x_test has not been modified by attack and classifier
    #     self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)
    #
    #     # Clean-up
    #     k.clear_session()
    #
    # def test_pytorch_mnist_L2(self):
    #     """
    #     Third test with the PyTorchClassifier.
    #     :return:
    #     """
    #     x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
    #     x_test_original = x_test.copy()
    #
    #     # Build PyTorchClassifier
    #     ptc = get_image_classifier_pt(from_logits=True)
    #
    #     # First attack
    #     cl2m = CarliniL2Method(classifier=ptc, targeted=True, max_iter=10, verbose=False)
    #     params = {"y": random_targets(self.y_test_mnist, ptc.nb_classes)}
    #     x_test_adv = cl2m.generate(x_test, **params)
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target == y_pred_adv).any())
    #     logger.info("CW2 Success Rate: %.2f", (sum(target == y_pred_adv) / float(len(target))))
    #
    #     # Second attack
    #     cl2m = CarliniL2Method(classifier=ptc, targeted=False, max_iter=10, verbose=False)
    #     x_test_adv = cl2m.generate(x_test)
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target != y_pred_adv).any())
    #     logger.info("CW2 Success Rate: %.2f", (sum(target != y_pred_adv) / float(len(target))))
    #
    #     # Check that x_test has not been modified by attack and classifier
    #     self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail_L2(self):
        backend_test_classifier_type_check_fail(CarliniL2Method, [BaseEstimator, ClassGradientsMixin])

    # def test_keras_iris_clipped_L2(self):
    #     classifier = get_tabular_classifier_kr()
    #     attack = CarliniL2Method(classifier, targeted=False, max_iter=10, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))
    #
    # def test_keras_iris_unbounded_L2(self):
    #     classifier = get_tabular_classifier_kr()
    #
    #     # Recreate a classifier without clip values
    #     classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
    #     attack = CarliniL2Method(classifier, targeted=False, max_iter=10, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))

    # def test_tensorflow_iris_L2(self):
    #     classifier, _ = get_tabular_classifier_tf()
    #
    #     # Test untargeted attack
    #     attack = CarliniL2Method(classifier, targeted=False, max_iter=10, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))
    #
    #     # Test targeted attack
    #     targets = random_targets(self.y_test_iris, nb_classes=3)
    #     attack = CarliniL2Method(classifier, targeted=True, max_iter=10, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
    #     accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Success rate of targeted C&W on Iris: %.2f%%", (accuracy * 100))

    # def test_pytorch_iris_L2(self):
    #     classifier = get_tabular_classifier_pt()
    #     attack = CarliniL2Method(classifier, targeted=False, max_iter=10, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))

    # def test_scikitlearn_L2(self):
    #     from sklearn.linear_model import LogisticRegression
    #     from sklearn.svm import SVC, LinearSVC
    #
    #     from art.estimators.classification.scikitlearn import SklearnClassifier
    #
    #     scikitlearn_test_cases = [
    #         LogisticRegression(solver="lbfgs", multi_class="auto"),
    #         SVC(gamma="auto"),
    #         LinearSVC(),
    #     ]
    #
    #     x_test_original = self.x_test_iris.copy()
    #
    #     for model in scikitlearn_test_cases:
    #         classifier = SklearnClassifier(model=model, clip_values=(0, 1))
    #         classifier.fit(x=self.x_test_iris, y=self.y_test_iris)
    #
    #         # Test untargeted attack
    #         attack = CarliniL2Method(classifier, targeted=False, max_iter=2, verbose=False)
    #         x_test_adv = attack.generate(self.x_test_iris)
    #         self.assertFalse((self.x_test_iris == x_test_adv).all())
    #         self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #         self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #         predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #         self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #         accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #         logger.info(
    #             "Accuracy of " + classifier.__class__.__name__ + " on Iris with C&W adversarial examples: " "%.2f%%",
    #             (accuracy * 100),
    #         )
    #
    #         # Test targeted attack
    #         targets = random_targets(self.y_test_iris, nb_classes=3)
    #         attack = CarliniL2Method(classifier, targeted=True, max_iter=2, verbose=False)
    #         x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #         self.assertFalse((self.x_test_iris == x_test_adv).all())
    #         self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #         self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #         predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #         self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
    #         accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #         logger.info(
    #             "Success rate of " + classifier.__class__.__name__ + " on targeted C&W on Iris: %.2f%%",
    #             (accuracy * 100),
    #         )
    #
    #         # Check that x_test has not been modified by attack and classifier
    #         self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)

    """
    A unittest class for testing the Carlini LInf attack.
    """

    def test_tensorflow_failure_attack_LInf(self):
        """
        Test the corner case when attack is failed.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf(from_logits=True)

        # Failure attack
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=0, learning_rate=0, eps=0.5, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = clinfm.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        self.assertTrue(np.allclose(self.x_test_mnist, x_test_adv, atol=1e-3))

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_tensorflow_mnist_LInf(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf(from_logits=True)

        # First attack
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=10, eps=0.5, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = clinfm.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("CW0 Target: %s", target)
        logger.debug("CW0 Actual: %s", y_pred_adv)
        logger.info("CW0 Success Rate: %.2f", (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack, no batching
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=False, max_iter=10, eps=0.5, batch_size=1, verbose=False)
        x_test_adv = clinfm.generate(self.x_test_mnist)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("CW0 Target: %s", target)
        logger.debug("CW0 Actual: %s", y_pred_adv)
        logger.info("CW0 Success Rate: %.2f", (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up session
        if sess is not None:
            sess.close()

    # @unittest.skipIf(
    #     not (int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3),
    #     reason="Keras 2.3 or later or TensorFlow-Keras required to support selected combination of loss "
    #     "function and logits.",
    # )
    # def test_keras_mnist_LInf(self):
    #     """
    #     Second test with the KerasClassifier.
    #     :return:
    #     """
    #     # Build KerasClassifier
    #     krc = get_image_classifier_kr(from_logits=True)
    #
    #     # First attack
    #     clinfm = CarliniLInfMethod(classifier=krc, targeted=True, max_iter=10, eps=0.5, verbose=False)
    #     params = {"y": random_targets(self.y_test_mnist, krc.nb_classes)}
    #     x_test_adv = clinfm.generate(self.x_test_mnist, **params)
    #     self.assertFalse((self.x_test_mnist == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.000001)
    #     self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
    #     logger.debug("CW0 Target: %s", target)
    #     logger.debug("CW0 Actual: %s", y_pred_adv)
    #     logger.info("CW0 Success Rate: %.2f", (np.sum(target == y_pred_adv) / float(len(target))))
    #     self.assertTrue((target == y_pred_adv).any())
    #
    #     # Second attack
    #     clinfm = CarliniLInfMethod(classifier=krc, targeted=False, max_iter=10, eps=0.5, verbose=False)
    #     x_test_adv = clinfm.generate(self.x_test_mnist)
    #     self.assertLessEqual(np.amax(x_test_adv), 1.000001)
    #     self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
    #     logger.debug("CW0 Target: %s", target)
    #     logger.debug("CW0 Actual: %s", y_pred_adv)
    #     logger.info("CW0 Success Rate: %.2f", (np.sum(target != y_pred_adv) / float(len(target))))
    #     self.assertTrue((target != y_pred_adv).any())
    #
    #     # Clean-up
    #     k.clear_session()
    #
    # def test_pytorch_mnist_LInf(self):
    #     """
    #     Third test with the PyTorchClassifier.
    #     :return:
    #     """
    #     x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
    #
    #     # Build PyTorchClassifier
    #     ptc = get_image_classifier_pt(from_logits=True)
    #
    #     # First attack
    #     clinfm = CarliniLInfMethod(classifier=ptc, targeted=True, max_iter=10, eps=0.5, verbose=False)
    #     params = {"y": random_targets(self.y_test_mnist, ptc.nb_classes)}
    #     x_test_adv = clinfm.generate(x_test, **params)
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0 + 1e-6)
    #     self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target == y_pred_adv).any())
    #
    #     # Second attack
    #     clinfm = CarliniLInfMethod(classifier=ptc, targeted=False, max_iter=10, eps=0.5, verbose=False)
    #     x_test_adv = clinfm.generate(x_test)
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0 + 1e-6)
    #     self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
    #
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target != y_pred_adv).any())

    def test_classifier_type_check_fail_LInf(self):
        backend_test_classifier_type_check_fail(CarliniLInfMethod, [BaseEstimator, ClassGradientsMixin])

    # def test_keras_iris_clipped_LInf(self):
    #     classifier = get_tabular_classifier_kr()
    #     attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))
    #
    # def test_keras_iris_unbounded_LInf(self):
    #     classifier = get_tabular_classifier_kr()
    #
    #     # Recreate a classifier without clip values
    #     classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
    #     attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=1, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))

    # def test_tensorflow_iris_LInf(self):
    #     classifier, _ = get_tabular_classifier_tf()
    #
    #     # Test untargeted attack
    #     attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))
    #
    #     # Test targeted attack
    #     targets = random_targets(self.y_test_iris, nb_classes=3)
    #     attack = CarliniLInfMethod(classifier, targeted=True, max_iter=10, eps=0.5, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
    #     accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Success rate of targeted C&W on Iris: %.2f%%", (accuracy * 100))

    # def test_pytorch_iris_LInf(self):
    #     classifier = get_tabular_classifier_pt()
    #     attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5, verbose=False)
    #     x_test_adv = attack.generate(self.x_test_iris.astype(np.float32))
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #     self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #     predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #     accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with C&W adversarial examples: %.2f%%", (accuracy * 100))

    # def test_scikitlearn_LInf(self):
    #     from sklearn.linear_model import LogisticRegression
    #     from sklearn.svm import SVC, LinearSVC
    #
    #     from art.estimators.classification.scikitlearn import SklearnClassifier
    #
    #     scikitlearn_test_cases = [
    #         LogisticRegression(solver="lbfgs", multi_class="auto"),
    #         SVC(gamma="auto"),
    #         LinearSVC(),
    #     ]
    #
    #     x_test_original = self.x_test_iris.copy()
    #
    #     for model in scikitlearn_test_cases:
    #         classifier = SklearnClassifier(model=model, clip_values=(0, 1))
    #         classifier.fit(x=self.x_test_iris, y=self.y_test_iris)
    #
    #         # Test untargeted attack
    #         attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5, verbose=False)
    #         x_test_adv = attack.generate(self.x_test_iris)
    #         self.assertFalse((self.x_test_iris == x_test_adv).all())
    #         self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #         self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #         predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #         self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
    #         accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #         logger.info(
    #             "Accuracy of " + classifier.__class__.__name__ + " on Iris with C&W adversarial examples: " "%.2f%%",
    #             (accuracy * 100),
    #         )
    #
    #         # Test targeted attack
    #         targets = random_targets(self.y_test_iris, nb_classes=3)
    #         attack = CarliniLInfMethod(classifier, targeted=True, max_iter=10, eps=0.5, verbose=False)
    #         x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #         self.assertFalse((self.x_test_iris == x_test_adv).all())
    #         self.assertLessEqual(np.amax(x_test_adv), 1.0)
    #         self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
    #
    #         predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #         self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
    #         accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #         logger.info(
    #             "Success rate of " + classifier.__class__.__name__ + " on targeted C&W on Iris: %.2f%%",
    #             (accuracy * 100),
    #         )
    #
    #         # Check that x_test has not been modified by attack and classifier
    #         self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)


if __name__ == "__main__":
    unittest.main()
