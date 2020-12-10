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

import numpy as np
import pytest

from art.attacks.evasion.universal_perturbation import UniversalPerturbation

from tests.utils import ARTTestException
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 500
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


def test_image(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_mnist_subset

        x_test_original = x_test.copy()

        # Build TensorFlowClassifier
        estimator, _ = image_dl_estimator()

        # Attack
        up = UniversalPerturbation(estimator, max_iter=1, attacker="newtonfool", attacker_params={"max_iter": 5})
        x_train_adv = up.generate(x_train)
        assert (up.fooling_rate >= 0.2) or not up.converged

        x_test_adv = x_test + up.noise
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_adv)

        train_y_pred = np.argmax(estimator.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(estimator.predict(x_test_adv), axis=1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, np.argmax(y_test, axis=1), test_y_pred)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, np.argmax(y_train, axis=1), train_y_pred
        )

        # Check that x_test has not been modified by attack and classifier
        np.testing.assert_array_almost_equal(float(np.max(np.abs(x_test_original - x_test))), 0, decimal=5)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("clipped", [True, False])
def test_tabular(art_warning, get_iris_dataset, tabular_dl_estimator, clipped):
    try:
        (_, _), (x_test, y_test) = get_iris_dataset

        estimator = tabular_dl_estimator(clipped)

        # Test untargeted attack
        attack_params = {"max_iter": 1, "attacker": "ead", "attacker_params": {"max_iter": 5, "targeted": False}}
        attack = UniversalPerturbation(estimator)
        attack.set_params(**attack_params)
        x_test_iris_adv = attack.generate(x_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_iris_adv)

        if clipped:
            np.testing.assert_array_less(x_test_iris_adv, 1)
            # Note: the np version of the assert doesn't seem to pass
            # np.testing.assert_array_less(0, x_test_iris_adv)
            assert bool((x_test_iris_adv >= 0).all())

        preds_adv = np.argmax(estimator.predict(x_test_iris_adv), axis=1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, np.argmax(y_test, axis=1), preds_adv)
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on Iris with universal adversarial examples: %.2f%%", (acc * 100))
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(UniversalPerturbation, [BaseEstimator, ClassifierMixin])

# from __future__ import absolute_import, division, print_function, unicode_literals
#
# import logging
# import unittest
#
# import numpy as np
#
# from art.attacks.evasion.universal_perturbation import UniversalPerturbation
# from art.estimators.classification.classifier import ClassifierMixin
# from art.estimators.classification.keras import KerasClassifier
# from art.estimators.estimator import BaseEstimator
# from tests.attacks.utils import backend_test_classifier_type_check_fail
# from tests.utils import (
#     TestBase,
#     get_image_classifier_kr,
#     get_image_classifier_pt,
#     get_image_classifier_tf,
#     get_tabular_classifier_kr,
#     get_tabular_classifier_pt,
#     get_tabular_classifier_tf,
# )

# logger = logging.getLogger(__name__)
#
#
# class TestUniversalPerturbation(TestBase):
#     """
#     A unittest class for testing the UniversalPerturbation attack.
#     """
#
#     @classmethod
#     def setUpClass(cls):
#         super().setUpClass()
#
#         cls.n_train = 500
#         cls.n_test = 10
#         cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
#         cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
#         cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
#         cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]
#
#     def test_3_tensorflow_mnist(self):
#         """
#         First test with the TensorFlowClassifier.
#         :return:
#         """
#         x_test_original = self.x_test_mnist.copy()
#
#         # Build TensorFlowClassifier
#         tfc, sess = get_image_classifier_tf()
#
#         # Attack
#         up = UniversalPerturbation(tfc, max_iter=1, attacker="newtonfool", attacker_params={"max_iter": 5})
#         x_train_adv = up.generate(self.x_train_mnist)
#         self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)
#
#         x_test_adv = self.x_test_mnist + up.noise
#         self.assertFalse((self.x_test_mnist == x_test_adv).all())
#
#         train_y_pred = np.argmax(tfc.predict(x_train_adv), axis=1)
#         test_y_pred = np.argmax(tfc.predict(x_test_adv), axis=1)
#         self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
#         self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
#
#         # Check that x_test has not been modified by attack and classifier
#         self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)
#
#     def test_8_keras_mnist(self):
#         """
#         Second test with the KerasClassifier.
#         :return:
#         """
#         x_test_original = self.x_test_mnist.copy()
#
#         # Build KerasClassifier
#         krc = get_image_classifier_kr()
#
#         # Attack
#         up = UniversalPerturbation(krc, max_iter=1, attacker="ead", attacker_params={"max_iter": 2, "targeted": False})
#         x_train_adv = up.generate(self.x_train_mnist)
#         self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)
#
#         x_test_adv = self.x_test_mnist + up.noise
#         self.assertFalse((self.x_test_mnist == x_test_adv).all())
#
#         train_y_pred = np.argmax(krc.predict(x_train_adv), axis=1)
#         test_y_pred = np.argmax(krc.predict(x_test_adv), axis=1)
#         self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
#         self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
#
#         # Check that x_test has not been modified by attack and classifier
#         self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)
#
#     def test_5_pytorch_mnist(self):
#         """
#         Third test with the PyTorchClassifier.
#         :return:
#         """
#         x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
#         x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
#         x_test_original = x_test_mnist.copy()
#
#         # Build PyTorchClassifier
#         ptc = get_image_classifier_pt()
#
#         # Attack
#         up = UniversalPerturbation(ptc, max_iter=1, attacker="newtonfool", attacker_params={"max_iter": 5})
#         x_train_mnist_adv = up.generate(x_train_mnist)
#         self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)
#
#         x_test_mnist_adv = x_test_mnist + up.noise
#         self.assertFalse((x_test_mnist == x_test_mnist_adv).all())
#
#         train_y_pred = np.argmax(ptc.predict(x_train_mnist_adv), axis=1)
#         test_y_pred = np.argmax(ptc.predict(x_test_mnist_adv), axis=1)
#         self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
#         self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
#
#         # Check that x_test has not been modified by attack and classifier
#         self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=0.00001)
#
#     def test_6_keras_iris_clipped(self):
#         classifier = get_tabular_classifier_kr()
#
#         # Test untargeted attack
#         attack_params = {"max_iter": 1, "attacker": "newtonfool", "attacker_params": {"max_iter": 5}}
#         attack = UniversalPerturbation(classifier)
#         attack.set_params(**attack_params)
#         x_test_iris_adv = attack.generate(self.x_test_iris)
#         self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
#         self.assertTrue((x_test_iris_adv <= 1).all())
#         self.assertTrue((x_test_iris_adv >= 0).all())
#
#         preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
#         self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
#         acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
#         logger.info("Accuracy on Iris with universal adversarial examples: %.2f%%", (acc * 100))
#
#     def test_7_keras_iris_unbounded(self):
#         classifier = get_tabular_classifier_kr()
#
#         # Recreate a classifier without clip values
#         classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
#         attack_params = {"max_iter": 1, "attacker": "newtonfool", "attacker_params": {"max_iter": 5}}
#         attack = UniversalPerturbation(classifier)
#         attack.set_params(**attack_params)
#         x_test_iris_adv = attack.generate(self.x_test_iris)
#         self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
#
#         preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
#         self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
#         acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
#         logger.info("Accuracy on Iris with universal adversarial examples: %.2f%%", (acc * 100))
#
#     def test_2_tensorflow_iris(self):
#         classifier, _ = get_tabular_classifier_tf()
#
#         # Test untargeted attack
#         attack_params = {"max_iter": 1, "attacker": "ead", "attacker_params": {"max_iter": 5, "targeted": False}}
#         attack = UniversalPerturbation(classifier)
#         attack.set_params(**attack_params)
#         x_test_iris_adv = attack.generate(self.x_test_iris)
#         self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
#         self.assertTrue((x_test_iris_adv <= 1).all())
#         self.assertTrue((x_test_iris_adv >= 0).all())
#
#         preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
#         self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
#         acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
#         logger.info("Accuracy on Iris with universal adversarial examples: %.2f%%", (acc * 100))
#
#     def test_4_pytorch_iris(self):
#         classifier = get_tabular_classifier_pt()
#
#         attack_params = {"max_iter": 1, "attacker": "ead", "attacker_params": {"max_iter": 5, "targeted": False}}
#         attack = UniversalPerturbation(classifier)
#         attack.set_params(**attack_params)
#         x_test_iris_adv = attack.generate(self.x_test_iris)
#         self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
#         self.assertTrue((x_test_iris_adv <= 1).all())
#         self.assertTrue((x_test_iris_adv >= 0).all())
#
#         preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
#         self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
#         acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
#         logger.info("Accuracy on Iris with universal adversarial examples: %.2f%%", (acc * 100))
#
#     def test_1_classifier_type_check_fail(self):
#         backend_test_classifier_type_check_fail(UniversalPerturbation, [BaseEstimator, ClassifierMixin])
#
#
# if __name__ == "__main__":
#     unittest.main()
