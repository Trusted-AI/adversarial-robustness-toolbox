# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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

import pytest
import logging
import numpy as np

from art.defences.trainer import AdversarialTrainerFBFPyTorch
from tests.utils import get_image_classifier_hf

HF_MODEL_SIZE = "SMALL"


@pytest.fixture()
def get_adv_trainer(framework, image_dl_estimator):
    def _get_adv_trainer():

        if framework == "keras":
            trainer = None
        if framework in ["tensorflow", "tensorflow2v1"]:
            trainer = None
        if framework == "pytorch":
            classifier, _ = image_dl_estimator()
            trainer = AdversarialTrainerFBFPyTorch(classifier, eps=0.05)
        if framework == "scikitlearn":
            trainer = None
        if framework == "huggingface":
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
                    optimizer=optimizer,
                    input_shape=(3, 224, 224),
                    nb_classes=10,
                    processor=None,
                )
            elif HF_MODEL_SIZE == "SMALL":
                classifier = get_image_classifier_hf(from_logits=True)
            else:
                raise ValueError("HF_MODEL_SIZE must be either SMALL or LARGE")

            trainer = AdversarialTrainerFBFPyTorch(classifier, eps=0.05)

        return trainer

    return _get_adv_trainer


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 100
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.only_with_platform("huggingface")
def test_adversarial_trainer_fbf_huggingface_fit_and_predict(
    get_adv_trainer, get_default_cifar10_subset, fix_get_mnist_subset
):

    if HF_MODEL_SIZE == "LARGE":
        (x_train, y_train), (x_test, y_test) = get_default_cifar10_subset
        import torch

        x_train = x_train[0:100]
        y_train = y_train[0:100]
        upsampler = torch.nn.Upsample(scale_factor=7, mode="nearest")
        x_train = np.rollaxis(x_train, 3, 1)
        x_test = np.rollaxis(x_test, 3, 1)
    else:
        (x_train, y_train, x_test, y_test) = fix_get_mnist_subset

    if HF_MODEL_SIZE == "LARGE":
        x_train = np.float32(upsampler(torch.from_numpy(x_train)).cpu().numpy())
        x_test = np.float32(upsampler(torch.from_numpy(x_test)).cpu().numpy())

    x_test_original = x_test.copy()

    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return

    predictions = np.argmax(trainer.predict(x_test), axis=1)
    accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / x_test.shape[0]

    trainer.fit(x_train, y_train, nb_epochs=20, validation_data=(x_test, y_test))
    predictions_new = np.argmax(trainer.predict(x_test), axis=1)
    accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / x_test.shape[0]

    np.testing.assert_array_almost_equal(
        float(np.mean(x_test_original - x_test)),
        0.0,
        decimal=4,
    )

    if HF_MODEL_SIZE == "SMALL":
        # NB, differs from pytorch due to issiue number #2227. Here we use logits for Huggingface.
        assert accuracy == 0.32
        # Different platforms gave marginally different results
        assert 0.65 <= accuracy_new <= 0.67


@pytest.mark.skip_framework("tensorflow", "keras", "scikitlearn", "mxnet", "kerastf", "huggingface")
def test_adversarial_trainer_fbf_pytorch_fit_and_predict(get_adv_trainer, fix_get_mnist_subset):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()

    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return

    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]

    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]

    np.testing.assert_array_almost_equal(
        float(np.mean(x_test_mnist_original - x_test_mnist)),
        0.0,
        decimal=4,
    )

    assert accuracy == 0.32
    assert accuracy_new == 0.63

    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))


@pytest.mark.skip_framework("tensorflow", "keras", "scikitlearn", "mxnet", "kerastf", "huggingface")
def test_adversarial_trainer_fbf_pytorch_fit_generator_and_predict(
    get_adv_trainer, fix_get_mnist_subset, image_data_generator
):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()

    generator = image_data_generator()

    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return

    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]

    trainer.fit_generator(generator=generator, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]

    np.testing.assert_array_almost_equal(
        float(np.mean(x_test_mnist_original - x_test_mnist)),
        0.0,
        decimal=4,
    )

    assert accuracy == 0.32
    assert accuracy_new > 0.2
