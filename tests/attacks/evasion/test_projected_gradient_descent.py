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
import pytest

import numpy as np

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
    ProjectedGradientDescentCommon,
)
from art.estimators.classification import KerasClassifier, PyTorchClassifier, TensorFlowV2Classifier
from art.estimators.hugging_face import HuggingFaceClassifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.utils import get_labels_np_array, random_targets
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import (
    get_image_classifier_kr,
    get_image_classifier_pt,
    get_image_classifier_tf,
    get_image_classifier_hf,
    get_tabular_classifier_kr,
    get_tabular_classifier_pt,
    get_tabular_classifier_tf,
    get_tabular_classifier_hf,
    master_seed,
)

logger = logging.getLogger(__name__)
HF_MODEL_SIZE = "SMALL"


@pytest.fixture()
def get_mnist_classifier(framework, image_dl_estimator):
    def _get_classifier():
        if framework == "pytorch":
            import torch

            # NB, uses CROSSENTROPYLOSS thus should use logits in output
            classifier = get_image_classifier_pt(from_logits=True)

        elif framework == "tensorflow2":
            classifier, _ = get_image_classifier_tf()
        elif framework in ("keras", "kerastf"):
            classifier = get_image_classifier_kr()
        elif framework == "huggingface":
            if HF_MODEL_SIZE == "LARGE":
                import transformers
                import torch
                from art.estimators.hugging_face import HuggingFaceClassifier

                model = transformers.AutoModelForImageClassification.from_pretrained(
                    "facebook/deit-tiny-patch16-224", ignore_mismatched_sizes=True, num_labels=10
                )

                print("num of parameters is ", sum(p.numel() for p in model.parameters() if p.requires_grad))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                classifier = HuggingFaceClassifier(
                    model,
                    loss=torch.nn.CrossEntropyLoss(),
                    input_shape=(3, 224, 224),
                    nb_classes=10,
                    optimizer=optimizer,
                    processor=None,
                )
            elif HF_MODEL_SIZE == "SMALL":
                classifier = get_image_classifier_hf(from_logits=True)
            else:
                raise ValueError("HF_MODEL_SIZE must be either SMALL or LARGE")
        else:
            classifier = None

        return classifier

    return _get_classifier


@pytest.fixture()
def get_tabular_classifier(framework, image_dl_estimator):
    def _get_classifier():
        if framework == "pytorch":
            import torch

            # NB, uses CROSSENTROPYLOSS thus should use logits in output
            classifier = get_tabular_classifier_pt()

        elif framework == "tensorflow2":
            classifier, _ = get_tabular_classifier_tf()
        elif framework in ("keras", "kerastf"):
            classifier = get_tabular_classifier_kr()
        elif framework == "huggingface":
            if HF_MODEL_SIZE == "LARGE":
                import transformers
                import torch
                from art.estimators.hugging_face import HuggingFaceClassifier

                model = transformers.AutoModelForImageClassification.from_pretrained(
                    "facebook/deit-tiny-patch16-224", ignore_mismatched_sizes=True, num_labels=10  # takes 3 min
                )

                print("num of parameters is ", sum(p.numel() for p in model.parameters() if p.requires_grad))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                classifier = HuggingFaceClassifier(
                    model,
                    loss=torch.nn.CrossEntropyLoss(),
                    input_shape=(3, 224, 224),
                    nb_classes=10,
                    optimizer=optimizer,
                    processor=None,
                )
            elif HF_MODEL_SIZE == "SMALL":
                classifier = get_tabular_classifier_hf()
            else:
                raise ValueError("HF_MODEL_SIZE must be either SMALL or LARGE")
        else:
            classifier = None

        return classifier

    return _get_classifier


class TestPGD:
    @pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf", "scikitlearn", "huggingface")
    def test_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(ProjectedGradientDescent, [BaseEstimator, LossGradientsMixin])

    @pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf", "huggingface")
    def test_check_params(self, get_mnist_classifier):
        classifier = get_mnist_classifier()

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, norm=-1)

        with pytest.raises(TypeError):
            _ = ProjectedGradientDescentCommon(classifier, eps="1", eps_step=0.1)

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, eps=-1)

        with pytest.raises(TypeError):
            _ = ProjectedGradientDescentCommon(classifier, eps=np.array([-1]))

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, eps_step=-1)

        with pytest.raises(TypeError):
            _ = ProjectedGradientDescentCommon(classifier, eps_step=np.array([-1]))

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, eps=np.array([1.0, 1.0]), eps_step=np.array([1.0]))

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, targeted="False")

        with pytest.raises(TypeError):
            _ = ProjectedGradientDescentCommon(classifier, num_random_init="1")

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, num_random_init=-1)

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, batch_size=-1)

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, max_iter=-1)

        with pytest.raises(ValueError):
            _ = ProjectedGradientDescentCommon(classifier, verbose="False")

    @pytest.mark.only_with_platform("pytorch", "tensorflow2")
    def test_mnist(self, get_default_mnist_subset, get_mnist_classifier, framework):
        classifier = get_mnist_classifier()
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        scores = get_labels_np_array(classifier.predict(x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train_mnist, axis=1)) / y_train_mnist.shape[0]
        logger.info("[%s MNIST] Accuracy on training set: %.2f%%" % (framework, acc * 100))

        scores = get_labels_np_array(classifier.predict(x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
        logger.info("[%s, MNIST] Accuracy on test set: %.2f%%" % (framework, acc * 100))

        self._test_backend_mnist(classifier, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

        # Original tests only run this for pytorch
        if framework in ["pytorch", "tensorflow2"]:
            # Test with clip values of array type
            classifier.set_params(clip_values=(np.zeros_like(x_test_mnist[0]), np.ones_like(x_test_mnist[0])))
            self._test_backend_mnist(classifier, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

            classifier.set_params(clip_values=(np.zeros_like(x_test_mnist[0][0]), np.ones_like(x_test_mnist[0][0])))
            self._test_backend_mnist(classifier, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

            classifier.set_params(
                clip_values=(np.zeros_like(x_test_mnist[0][0][0]), np.ones_like(x_test_mnist[0][0][0]))
            )
            self._test_backend_mnist(classifier, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

    # Original tests ran keras tests in this pattern
    @pytest.mark.only_with_platform("keras", "kerastf")
    def test_9a_keras_mnist(self, get_default_mnist_subset, get_mnist_classifier, framework):
        # TODO: Raise bugreport. Keras tests fail if using a larger batch.

        classifier = get_mnist_classifier()
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
        n_train = 10
        n_test = 10

        x_train_mnist = x_train_mnist[0:n_train]
        y_train_mnist = y_train_mnist[0:n_train]
        x_test_mnist = x_test_mnist[0:n_test]
        y_test_mnist = y_test_mnist[0:n_test]

        scores = classifier._model.evaluate(x_train_mnist, y_train_mnist)
        logger.info("[Keras, MNIST] Accuracy on training set: %.2f%%", scores[1] * 100)
        scores = classifier._model.evaluate(x_test_mnist, y_test_mnist)
        logger.info("[Keras, MNIST] Accuracy on test set: %.2f%%", scores[1] * 100)

        self._test_backend_mnist(classifier, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

    def _test_backend_mnist(self, classifier, x_train, y_train, x_test, y_test):
        x_test_original = x_test.copy()

        # Test PGD with np.inf norm
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        assert not ((x_train == x_train_adv).all())
        assert not ((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        assert not ((y_train == train_y_pred).all())
        assert not ((y_test == test_y_pred).all())

        acc = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", acc * 100)

        acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", acc * 100)

        # Test PGD with 3 random initialisations
        attack = ProjectedGradientDescent(classifier, num_random_init=3, verbose=False)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        assert not ((x_train == x_train_adv).all())
        assert not ((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        assert not ((y_train == train_y_pred).all())
        assert not ((y_test == test_y_pred).all())

        acc = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples with 3 random initialisations: %.2f%%", acc * 100)

        acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples with 3 random initialisations: %.2f%%", acc * 100)

        # Check that x_test has not been modified by attack and classifier
        assert np.allclose(float(np.max(np.abs(x_test_original - x_test))), 0.0)

        # Test the masking
        attack = ProjectedGradientDescent(classifier, num_random_init=1, verbose=False)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
        mask = mask.reshape(x_test.shape).astype(np.float32)

        x_test_adv = attack.generate(x_test, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - x_test)
        assert np.allclose(float(np.max(np.abs(mask_diff))), 0.0)

        # Test eps of array type 1
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)

        eps = np.ones(shape=x_test.shape) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        assert not ((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        assert not ((y_test == test_y_pred).all())

        # Test eps of array type 2
        eps = np.ones(shape=x_test.shape[1:]) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        assert not ((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        assert not ((y_test == test_y_pred).all())

        # Test eps of array type 3
        eps = np.ones(shape=x_test.shape[2:]) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        assert not ((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        assert not ((y_test == test_y_pred).all())

        # Test eps of array type 4
        eps = np.ones(shape=x_test.shape[3:]) * 1.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test)
        assert not ((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        assert not ((y_test == test_y_pred).all())

    @pytest.mark.only_with_platform("tensorflow2", "pytorch", "huggingface")
    def test_4_framework_mnist(self, get_mnist_classifier, get_default_mnist_subset):
        classifier = get_mnist_classifier()
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
        self._test_framework_vs_numpy(classifier, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

    def _test_framework_vs_numpy(self, classifier, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist):
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
        x_train_adv_np = attack_np.generate(x_train_mnist)
        x_test_adv_np = attack_np.generate(x_test_mnist)

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
        x_train_adv_fw = attack_fw.generate(x_train_mnist)
        x_test_adv_fw = attack_fw.generate(x_test_mnist)

        # Test
        assert np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

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
        x_train_adv_np = attack_np.generate(x_train_mnist)
        x_test_adv_np = attack_np.generate(x_test_mnist)

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
        x_train_adv_fw = attack_fw.generate(x_train_mnist)
        x_test_adv_fw = attack_fw.generate(x_test_mnist)

        # Test
        np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

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
        x_train_adv_np = attack_np.generate(x_train_mnist)
        x_test_adv_np = attack_np.generate(x_test_mnist)

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
        x_train_adv_fw = attack_fw.generate(x_train_mnist)
        x_test_adv_fw = attack_fw.generate(x_test_mnist)

        # Test
        np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

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
        x_train_adv_np = attack_np.generate(x_train_mnist, y_train_mnist)
        x_test_adv_np = attack_np.generate(x_test_mnist, y_test_mnist)

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
        x_train_adv_fw = attack_fw.generate(x_train_mnist, y_train_mnist)
        x_test_adv_fw = attack_fw.generate(x_test_mnist, y_test_mnist)

        # Test
        np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

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
        x_train_adv_np = attack_np.generate(x_train_mnist)
        x_test_adv_np = attack_np.generate(x_test_mnist)

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
        x_train_adv_fw = attack_fw.generate(x_train_mnist)
        x_test_adv_fw = attack_fw.generate(x_test_mnist)

        # Test
        np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

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
        x_train_adv_np = attack_np.generate(x_train_mnist)
        x_test_adv_np = attack_np.generate(x_test_mnist)

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
        x_train_adv_fw = attack_fw.generate(x_train_mnist)
        x_test_adv_fw = attack_fw.generate(x_test_mnist)

        # Test
        np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

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

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_train_mnist.shape))
        mask = mask.reshape(x_train_mnist.shape).astype(np.float32)
        x_train_adv_np = attack_np.generate(x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test_mnist.shape))
        mask = mask.reshape(x_test_mnist.shape).astype(np.float32)
        x_test_adv_np = attack_np.generate(x_test_mnist, mask=mask)

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

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_train_mnist.shape))
        mask = mask.reshape(x_train_mnist.shape).astype(np.float32)
        x_train_adv_fw = attack_fw.generate(x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test_mnist.shape))
        mask = mask.reshape(x_test_mnist.shape).astype(np.float32)
        x_test_adv_fw = attack_fw.generate(x_test_mnist, mask=mask)

        # Test
        np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

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

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_train_mnist.shape[1:]))
        mask = mask.reshape(x_train_mnist.shape[1:]).astype(np.float32)
        x_train_adv_np = attack_np.generate(x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test_mnist.shape[1:]))
        mask = mask.reshape(x_test_mnist.shape[1:]).astype(np.float32)
        x_test_adv_np = attack_np.generate(x_test_mnist, mask=mask)

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

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_train_mnist.shape[1:]))
        mask = mask.reshape(x_train_mnist.shape[1:]).astype(np.float32)
        x_train_adv_fw = attack_fw.generate(x_train_mnist, mask=mask)

        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test_mnist.shape[1:]))
        mask = mask.reshape(x_test_mnist.shape[1:]).astype(np.float32)
        x_test_adv_fw = attack_fw.generate(x_test_mnist, mask=mask)

        # Test
        np.allclose(np.mean(x_train_adv_np - x_train_mnist), np.mean(x_train_adv_fw - x_train_mnist), atol=1e-06)
        np.allclose(np.mean(x_test_adv_np - x_test_mnist), np.mean(x_test_adv_fw - x_test_mnist), atol=1e-06)

    @pytest.mark.only_with_platform("pytorch", "huggingface", "tensorflow2", "keras", "kerastf")
    def test_iris_pt(self, get_iris_dataset, get_tabular_classifier):
        classifier = get_tabular_classifier()
        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # Test untargeted attack
        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(x_test_iris)
        assert not (x_test_iris == x_test_adv).all()
        assert (x_test_adv <= 1).all()
        assert (x_test_adv >= 0).all()

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert not (np.argmax(y_test_iris, axis=1) == preds_adv).all()
        acc = np.sum(preds_adv == np.argmax(y_test_iris, axis=1)) / y_test_iris.shape[0]
        logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))

        # Test targeted attack
        targets = random_targets(y_test_iris, nb_classes=3)
        attack = ProjectedGradientDescent(classifier, targeted=True, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
        x_test_adv = attack.generate(x_test_iris, **{"y": targets})
        assert not (x_test_iris == x_test_adv).all()
        assert (x_test_adv <= 1).all()
        assert (x_test_adv >= 0).all()

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert (np.argmax(targets, axis=1) == preds_adv).any()
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / y_test_iris.shape[0]
        logger.info("Success rate of targeted PGD on Iris: %.2f%%", (acc * 100))

    @pytest.mark.only_with_platform("pytorch", "huggingface", "tensorflow2", "keras", "kerastf")
    def test_iris_unbounded(self, framework, get_iris_dataset, get_tabular_classifier):
        classifier = get_tabular_classifier()
        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # Recreate a classifier without clip values
        if framework in ["keras", "kerastf"]:
            classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        elif framework == "pytorch":
            classifier = PyTorchClassifier(
                model=classifier.model,
                nb_classes=classifier.nb_classes,
                input_shape=classifier._input_shape,
                loss=classifier._loss,
                channels_first=True,
            )
        elif framework == "tensorflow2":
            classifier = TensorFlowV2Classifier(
                model=classifier._model,
                nb_classes=classifier.nb_classes,
                input_shape=classifier.input_shape,
                loss_object=classifier.loss_object,
                channels_first=True,
            )
        elif framework == "huggingface":
            classifier = HuggingFaceClassifier(
                model=classifier.model,
                nb_classes=classifier.nb_classes,
                loss=classifier.loss,
                input_shape=classifier._input_shape,
            )

        attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.2, max_iter=5, verbose=False)
        x_test_adv = attack.generate(x_test_iris)
        assert not (x_test_iris == x_test_adv).all()

        assert (x_test_adv > 1).any()
        assert (x_test_adv < 0).any()

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert not (np.argmax(y_test_iris, axis=1) == preds_adv).all()
        acc = np.sum(preds_adv == np.argmax(y_test_iris, axis=1)) / y_test_iris.shape[0]
        logger.info("Accuracy on Iris with PGD adversarial examples: %.2f%%", (acc * 100))

    @pytest.mark.only_with_platform("scikitlearn")
    def test_scikitlearn(self, get_iris_dataset):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.estimators.classification.scikitlearn import SklearnClassifier

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        scikitlearn_test_cases = [
            LogisticRegression(solver="lbfgs", multi_class="auto"),
            SVC(gamma="auto"),
            LinearSVC(),
        ]

        x_test_original = x_test_iris.copy()

        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=x_test_iris, y=y_test_iris)

            # Test untargeted attack
            attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, max_iter=5, verbose=False)
            x_test_adv = attack.generate(x_test_iris)
            assert not ((x_test_iris == x_test_adv).all())
            assert (x_test_adv <= 1).all()
            assert (x_test_adv >= 0).all()

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            assert not (np.argmax(y_test_iris, axis=1) == preds_adv).all()
            acc = np.sum(preds_adv == np.argmax(y_test_iris, axis=1)) / y_test_iris.shape[0]
            logger.info(
                "Accuracy of " + classifier.__class__.__name__ + " on Iris with PGD adversarial examples: " "%.2f%%",
                (acc * 100),
            )

            # Test targeted attack
            targets = random_targets(y_test_iris, nb_classes=3)
            attack = ProjectedGradientDescent(
                classifier, targeted=True, eps=1.0, eps_step=0.1, max_iter=5, verbose=False
            )
            x_test_adv = attack.generate(x_test_iris, **{"y": targets})
            assert not ((x_test_iris == x_test_adv).all())
            assert (x_test_adv <= 1).all()
            assert (x_test_adv >= 0).all()

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            assert (np.argmax(targets, axis=1) == preds_adv).any()
            acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / y_test_iris.shape[0]
            logger.info(
                "Success rate of " + classifier.__class__.__name__ + " on targeted PGD on Iris: %.2f%%", (acc * 100)
            )

            # Check that x_test has not been modified by attack and classifier
            assert np.allclose(float(np.max(np.abs(x_test_original - x_test_iris))), 0.0, atol=1e-06)


"""
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


    def test_check_params_pt(self):

        ptc = get_image_classifier_pt(from_logits=True)

        with self.assertRaises(TypeError):
            _ = ProjectedGradientDescent(ptc, eps=np.array([1, 1, 1]), eps_step=1)

        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, norm=0)

        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, eps=-1, eps_step=1)
        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, eps=np.array([-1, -1, -1]), eps_step=np.array([1, 1, 1]))

        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, eps=1, eps_step=-1)
        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, eps=np.array([1, 1, 1]), eps_step=np.array([-1, -1, -1]))

        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, targeted="true")

        with self.assertRaises(TypeError):
            _ = ProjectedGradientDescent(ptc, num_random_init=1.0)
        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, num_random_init=-1)

        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, batch_size=-1)

        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, max_iter=-1)

        with self.assertRaises(ValueError):
            _ = ProjectedGradientDescent(ptc, verbose="true")


if __name__ == "__main__":
    unittest.main()
"""
