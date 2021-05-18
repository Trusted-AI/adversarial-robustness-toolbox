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

import numpy as np
import tensorflow as tf

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from art.estimators.classification import KerasClassifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.utils import get_labels_np_array, random_targets
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import (
    TestBase,
    get_image_classifier_kr,
    get_image_classifier_pt,
    get_image_classifier_tf,
    get_tabular_classifier_kr,
    get_tabular_classifier_pt,
    get_tabular_classifier_tf,
    master_seed,
)

logger = logging.getLogger(__name__)


class TestPGD(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 10
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def test_9a_keras_mnist(self):
        classifier = get_image_classifier_kr()

        scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
        logger.info("[Keras, MNIST] Accuracy on training set: %.2f%%", scores[1] * 100)
        scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
        logger.info("[Keras, MNIST] Accuracy on test set: %.2f%%", scores[1] * 100)

        self._test_backend_mnist(
            classifier, self.x_train_mnist, self.y_train_mnist, self.x_test_mnist, self.y_test_mnist
        )

    def test_3_tensorflow_mnist(self):
        classifier, sess = get_image_classifier_tf()

        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on training set: %.2f%%", acc * 100)

        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on test set: %.2f%%", acc * 100)

        self._test_backend_mnist(
            classifier, self.x_train_mnist, self.y_train_mnist, self.x_test_mnist, self.y_test_mnist
        )

    def test_5_pytorch_mnist(self):
        x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        classifier = get_image_classifier_pt()

        scores = get_labels_np_array(classifier.predict(x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on training set: %.2f%%", acc * 100)

        scores = get_labels_np_array(classifier.predict(x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on test set: %.2f%%", acc * 100)

        self._test_backend_mnist(classifier, x_train_mnist, self.y_train_mnist, x_test_mnist, self.y_test_mnist)

        # Test with clip values of array type
        classifier.set_params(clip_values=(np.zeros_like(x_test_mnist[0]), np.ones_like(x_test_mnist[0])))
        self._test_backend_mnist(classifier, x_train_mnist, self.y_train_mnist, x_test_mnist, self.y_test_mnist)

        classifier.set_params(clip_values=(np.zeros_like(x_test_mnist[0][0]), np.ones_like(x_test_mnist[0][0])))
        self._test_backend_mnist(classifier, x_train_mnist, self.y_train_mnist, x_test_mnist, self.y_test_mnist)

        classifier.set_params(clip_values=(np.zeros_like(x_test_mnist[0][0][0]), np.ones_like(x_test_mnist[0][0][0])))
        self._test_backend_mnist(classifier, x_train_mnist, self.y_train_mnist, x_test_mnist, self.y_test_mnist)

    def _test_backend_mnist(self, classifier, x_train, y_train, x_test, y_test):
        x_test_original = x_test.copy()

        # Test PGD with np.inf norm
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", acc * 100)

        acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", acc * 100)

        # Test PGD with 3 random initialisations
        attack = ProjectedGradientDescent(classifier, num_random_init=3, verbose=False)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples with 3 random initialisations: %.2f%%", acc * 100)

        acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples with 3 random initialisations: %.2f%%", acc * 100)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # Test the masking
        attack = ProjectedGradientDescent(classifier, num_random_init=1, verbose=False)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
        mask = mask.reshape(x_test.shape).astype(np.float32)

        x_test_adv = attack.generate(x_test, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - x_test)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        # Test eps of array type 1
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)

        eps = np.ones(shape=x_test.shape) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == test_y_pred).all())

        # Test eps of array type 2
        eps = np.ones(shape=x_test.shape[1:]) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == test_y_pred).all())

        # Test eps of array type 3
        eps = np.ones(shape=x_test.shape[2:]) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == test_y_pred).all())

        # Test eps of array type 4
        eps = np.ones(shape=x_test.shape[3:]) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == test_y_pred).all())

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(ProjectedGradientDescent, [BaseEstimator, LossGradientsMixin])

    def test_8_keras_iris_clipped(self):
        classifier = get_tabular_classifier_kr()

        # Test untargeted attack
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))

        # Test targeted attack
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = ProjectedGradientDescent(classifier, targeted=True, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Success rate of targeted PGD on Iris: %.2f%%", (acc * 100))

    def test_keras_9_iris_unbounded(self):
        classifier = get_tabular_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.2, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv > 1).any())
        self.assertTrue((x_test_adv < 0).any())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))

    def test_2_tensorflow_iris(self):
        classifier, _ = get_tabular_classifier_tf()

        # Test untargeted attack
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))

        # Test targeted attack
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = ProjectedGradientDescent(classifier, targeted=True, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Success rate of targeted PGD on Iris: %.2f%%", (acc * 100))

    def test_4_pytorch_iris_pt(self):
        classifier = get_tabular_classifier_pt()

        # Test untargeted attack
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))

        # Test targeted attack
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = ProjectedGradientDescent(classifier, targeted=True, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Success rate of targeted PGD on Iris: %.2f%%", (acc * 100))

    def test_7_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.estimators.classification.scikitlearn import SklearnClassifier

        scikitlearn_test_cases = [
            LogisticRegression(solver="lbfgs", multi_class="auto"),
            SVC(gamma="auto"),
            LinearSVC(),
        ]

        x_test_original = self.x_test_iris.copy()

        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)

            # Test untargeted attack
            attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info(
                "Accuracy of " + classifier.__class__.__name__ + " on Iris with PGD adversarial examples: " "%.2f%%",
                (acc * 100),
            )

            # Test targeted attack
            targets = random_targets(self.y_test_iris, nb_classes=3)
            attack = ProjectedGradientDescent(
                classifier, targeted=True, eps=1.0, eps_step=0.1, max_iter=5, verbose=False
            )
            x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
            acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
            logger.info(
                "Success rate of " + classifier.__class__.__name__ + " on targeted PGD on Iris: %.2f%%", (acc * 100)
            )

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)

    @unittest.skipIf(tf.__version__[0] != "2", "")
    def test_4_framework_tensorflow_v2_mnist(self):
        classifier, _ = get_image_classifier_tf()
        self._test_framework_vs_numpy(classifier)

    def test_6_framework_pytorch_mnist(self):
        self.x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        self.x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)

        classifier = get_image_classifier_pt()
        self._test_framework_vs_numpy(classifier)

        self.x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        self.x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)

    def _test_framework_vs_numpy(self, classifier):
        # Test PGD with np.inf norm
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_np = attack_np.generate(self.x_train_mnist)
        x_test_adv_np = attack_np.generate(self.x_test_mnist)

        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )

        # Test PGD with L1 norm
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=1,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_np = attack_np.generate(self.x_train_mnist)
        x_test_adv_np = attack_np.generate(self.x_test_mnist)

        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=1,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )

        # Test PGD with L2 norm
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=2,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_np = attack_np.generate(self.x_train_mnist)
        x_test_adv_np = attack_np.generate(self.x_test_mnist)

        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=2,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )

        # Test PGD with True targeted
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=True,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_np = attack_np.generate(self.x_train_mnist, self.y_train_mnist)
        x_test_adv_np = attack_np.generate(self.x_test_mnist, self.y_test_mnist)

        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=True,
            num_random_init=0,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist, self.y_train_mnist)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist, self.y_test_mnist)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )

        # Test PGD with num_random_init=2
        master_seed(1234)
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=2,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_np = attack_np.generate(self.x_train_mnist)
        x_test_adv_np = attack_np.generate(self.x_test_mnist)

        master_seed(1234)
        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=2,
            batch_size=3,
            random_eps=False,
            verbose=False,
        )
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )

        # Test PGD with random_eps=True
        master_seed(1234)
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=True,
            verbose=False,
        )
        x_train_adv_np = attack_np.generate(self.x_train_mnist)
        x_test_adv_np = attack_np.generate(self.x_test_mnist)

        master_seed(1234)
        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=0,
            batch_size=3,
            random_eps=True,
            verbose=False,
        )
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )

        # Test the masking 1
        master_seed(1234)
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=1,
            batch_size=3,
            random_eps=True,
            verbose=False,
        )

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_train_mnist.shape))
        mask = mask.reshape(self.x_train_mnist.shape).astype(np.float32)
        x_train_adv_np = attack_np.generate(self.x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape).astype(np.float32)
        x_test_adv_np = attack_np.generate(self.x_test_mnist, mask=mask)

        master_seed(1234)
        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=1,
            batch_size=3,
            random_eps=True,
            verbose=False,
        )

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_train_mnist.shape))
        mask = mask.reshape(self.x_train_mnist.shape).astype(np.float32)
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape).astype(np.float32)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist, mask=mask)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )

        # Test the masking 2
        master_seed(1234)
        attack_np = ProjectedGradientDescentNumpy(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=1,
            batch_size=3,
            random_eps=True,
            verbose=False,
        )

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_train_mnist.shape[1:]))
        mask = mask.reshape(self.x_train_mnist.shape[1:]).astype(np.float32)
        x_train_adv_np = attack_np.generate(self.x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:]).astype(np.float32)
        x_test_adv_np = attack_np.generate(self.x_test_mnist, mask=mask)

        master_seed(1234)
        attack_fw = ProjectedGradientDescent(
            classifier,
            eps=1.0,
            eps_step=0.1,
            max_iter=5,
            norm=np.inf,
            targeted=False,
            num_random_init=1,
            batch_size=3,
            random_eps=True,
            verbose=False,
        )

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_train_mnist.shape[1:]))
        mask = mask.reshape(self.x_train_mnist.shape[1:]).astype(np.float32)
        x_train_adv_fw = attack_fw.generate(self.x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:]).astype(np.float32)
        x_test_adv_fw = attack_fw.generate(self.x_test_mnist, mask=mask)

        # Test
        self.assertAlmostEqual(
            np.mean(x_train_adv_np - self.x_train_mnist), np.mean(x_train_adv_fw - self.x_train_mnist), places=6
        )
        self.assertAlmostEqual(
            np.mean(x_test_adv_np - self.x_test_mnist), np.mean(x_test_adv_fw - self.x_test_mnist), places=6
        )


if __name__ == "__main__":
    unittest.main()
