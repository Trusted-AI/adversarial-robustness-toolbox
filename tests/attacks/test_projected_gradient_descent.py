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

from art.attacks import ProjectedGradientDescent
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import ProjectedGradientDescentNumpy
from art.classifiers import KerasClassifier
from art.classifiers.classifier import ClassifierGradients
from art.utils import get_labels_np_array, random_targets

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail

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

    # def test_keras_mnist(self):
    #     classifier = get_image_classifier_kr()
    #
    #     scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
    #     logger.info("[Keras, MNIST] Accuracy on training set: %.2f%%", scores[1] * 100)
    #     scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
    #     logger.info("[Keras, MNIST] Accuracy on test set: %.2f%%", scores[1] * 100)
    #
    #     self._test_backend_mnist(
    #         classifier, self.x_train_mnist, self.y_train_mnist, self.x_test_mnist, self.y_test_mnist
    #     )

    def test_tensorflow_mnist(self):
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

    # def test_pytorch_mnist(self):
    #     x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
    #     x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
    #     classifier = get_image_classifier_pt()
    #
    #     scores = get_labels_np_array(classifier.predict(x_train_mnist))
    #     acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
    #     logger.info("[PyTorch, MNIST] Accuracy on training set: %.2f%%", acc * 100)
    #
    #     scores = get_labels_np_array(classifier.predict(x_test_mnist))
    #     acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
    #     logger.info("[PyTorch, MNIST] Accuracy on test set: %.2f%%", acc * 100)
    #
    #     self._test_backend_mnist(classifier, x_train_mnist, self.y_train_mnist, x_test_mnist, self.y_test_mnist)

    def _test_backend_mnist(self, classifier, x_train, y_train, x_test, y_test):
        x_test_original = x_test.copy()

        # This is for testing the numpy version
        # Test PGD with np.inf norm
    #    attack = ProjectedGradientDescentNumpy(classifier, eps=1, eps_step=0.1, max_iter=10)
        #x_train_adv_np = attack.generate(x_train)
    #    x_test_adv_np = attack.generate(x_test)


        # This is for testing the framework-dependent version
        # Test PGD with np.inf norm
        attack = ProjectedGradientDescent(classifier, eps=1, eps_step=0.1, max_iter=10)
        #x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

    #    print(np.mean(x_test_adv_np), np.mean(x_test_adv))
        print(np.mean(x_test_adv))
        self.assertFalse(True)




        # self.assertFalse((x_train == x_train_adv).all())
        # self.assertFalse((x_test == x_test_adv).all())
        #
        # train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        # test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        #
        # self.assertFalse((y_train == train_y_pred).all())
        # self.assertFalse((y_test == test_y_pred).all())
        #
        # acc = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        # logger.info("Accuracy on adversarial train examples: %.2f%%", acc * 100)
        #
        # acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # logger.info("Accuracy on adversarial test examples: %.2f%%", acc * 100)
        #
        # # Test PGD with 3 random initialisations
        # attack = ProjectedGradientDescent(classifier, num_random_init=3)
        # x_train_adv = attack.generate(x_train)
        # x_test_adv = attack.generate(x_test)
        #
        # self.assertFalse((x_train == x_train_adv).all())
        # self.assertFalse((x_test == x_test_adv).all())
        #
        # train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        # test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        #
        # self.assertFalse((y_train == train_y_pred).all())
        # self.assertFalse((y_test == test_y_pred).all())
        #
        # acc = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        # logger.info("Accuracy on adversarial train examples with 3 random initialisations: %.2f%%", acc * 100)
        #
        # acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # logger.info("Accuracy on adversarial test examples with 3 random initialisations: %.2f%%", acc * 100)
        #
        # # Check that x_test has not been modified by attack and classifier
        # self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)






    # def test_classifier_type_check_fail(self):
    #     backend_test_classifier_type_check_fail(ProjectedGradientDescent, [ClassifierGradients])
    #
    # def test_keras_iris_clipped(self):
    #     classifier = get_tabular_classifier_kr()
    #
    #     # Test untargeted attack
    #     attack = ProjectedGradientDescent(classifier, eps=1, eps_step=0.1, max_iter=5)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1).all())
    #     self.assertTrue((x_test_adv >= 0).all())
    #
    #     preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
    #     acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))
    #
    #     # Test targeted attack
    #     targets = random_targets(self.y_test_iris, nb_classes=3)
    #     attack = ProjectedGradientDescent(classifier, targeted=True, eps=1, eps_step=0.1, max_iter=5)
    #     x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1).all())
    #     self.assertTrue((x_test_adv >= 0).all())
    #
    #     preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
    #     acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Success rate of targeted PGD on Iris: %.2f%%", (acc * 100))
    #
    # def test_keras_iris_unbounded(self):
    #     classifier = get_tabular_classifier_kr()
    #
    #     # Recreate a classifier without clip values
    #     classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
    #     attack = ProjectedGradientDescent(classifier, eps=1, eps_step=0.2, max_iter=5)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertTrue((x_test_adv > 1).any())
    #     self.assertTrue((x_test_adv < 0).any())
    #
    #     preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
    #     acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))
    #
    # def test_tensorflow_iris(self):
    #     classifier, _ = get_tabular_classifier_tf()
    #
    #     # Test untargeted attack
    #     attack = ProjectedGradientDescent(classifier, eps=1, eps_step=0.1, max_iter=5)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1).all())
    #     self.assertTrue((x_test_adv >= 0).all())
    #
    #     preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
    #     acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))
    #
    #     # Test targeted attack
    #     targets = random_targets(self.y_test_iris, nb_classes=3)
    #     attack = ProjectedGradientDescent(classifier, targeted=True, eps=1, eps_step=0.1, max_iter=5)
    #     x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1).all())
    #     self.assertTrue((x_test_adv >= 0).all())
    #
    #     preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
    #     acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Success rate of targeted PGD on Iris: %.2f%%", (acc * 100))
    #
    # def test_pytorch_iris_pt(self):
    #     classifier = get_tabular_classifier_pt()
    #
    #     # Test untargeted attack
    #     attack = ProjectedGradientDescent(classifier, eps=1, eps_step=0.1, max_iter=5)
    #     x_test_adv = attack.generate(self.x_test_iris)
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1).all())
    #     self.assertTrue((x_test_adv >= 0).all())
    #
    #     preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
    #     acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))
    #
    #     # Test targeted attack
    #     targets = random_targets(self.y_test_iris, nb_classes=3)
    #     attack = ProjectedGradientDescent(classifier, targeted=True, eps=1, eps_step=0.1, max_iter=5)
    #     x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #     self.assertFalse((self.x_test_iris == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1).all())
    #     self.assertTrue((x_test_adv >= 0).all())
    #
    #     preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #     self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
    #     acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #     logger.info("Success rate of targeted PGD on Iris: %.2f%%", (acc * 100))
    #
    # def test_scikitlearn(self):
    #     from sklearn.linear_model import LogisticRegression
    #     from sklearn.svm import SVC, LinearSVC
    #
    #     from art.classifiers.scikitlearn import SklearnClassifier
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
    #         attack = ProjectedGradientDescent(classifier, eps=1, eps_step=0.1, max_iter=5)
    #         x_test_adv = attack.generate(self.x_test_iris)
    #         self.assertFalse((self.x_test_iris == x_test_adv).all())
    #         self.assertTrue((x_test_adv <= 1).all())
    #         self.assertTrue((x_test_adv >= 0).all())
    #
    #         preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #         self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
    #         acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
    #         logger.info(
    #             "Accuracy of " + classifier.__class__.__name__ + " on Iris with PGD adversarial examples: " "%.2f%%",
    #             (acc * 100),
    #         )
    #
    #         # Test targeted attack
    #         targets = random_targets(self.y_test_iris, nb_classes=3)
    #         attack = ProjectedGradientDescent(classifier, targeted=True, eps=1, eps_step=0.1, max_iter=5)
    #         x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
    #         self.assertFalse((self.x_test_iris == x_test_adv).all())
    #         self.assertTrue((x_test_adv <= 1).all())
    #         self.assertTrue((x_test_adv >= 0).all())
    #
    #         preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    #         self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
    #         acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
    #         logger.info(
    #             "Success rate of " + classifier.__class__.__name__ + " on targeted PGD on Iris: %.2f%%", (acc * 100)
    #         )
    #
    #         # Check that x_test has not been modified by attack and classifier
    #         self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)


if __name__ == "__main__":
    unittest.main()
