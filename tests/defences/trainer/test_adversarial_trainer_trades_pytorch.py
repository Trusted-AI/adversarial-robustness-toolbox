# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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

from art.utils import load_dataset
from art.defences.trainer import AdversarialTrainerTRADESPyTorch
from art.attacks.evasion import ProjectedGradientDescent


@pytest.fixture()
def get_adv_trainer(framework, image_dl_estimator):
    def _get_adv_trainer():

        if framework == "keras":
            trainer = None
        if framework in ["tensorflow", "tensorflow2v1"]:
            trainer = None
        if framework == "pytorch":
            classifier, _ = image_dl_estimator(from_logits=True)
            attack = ProjectedGradientDescent(
                classifier,
                norm=np.inf,
                eps=0.3,
                eps_step=0.03,
                max_iter=20,
                targeted=False,
                num_random_init=1,
                batch_size=128,
                verbose=False,
            )
            trainer = AdversarialTrainerTRADESPyTorch(classifier, attack, beta=6.0)

        if framework == "huggingface":

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
                classifier = get_image_classifier_hf()

            attack = ProjectedGradientDescent(
                classifier,
                norm=np.inf,
                eps=0.3,
                eps_step=0.03,
                max_iter=5,
                targeted=False,
                num_random_init=1,
                batch_size=128,
                verbose=False,
            )

            trainer = AdversarialTrainerTRADESPyTorch(classifier, attack, beta=6.0)

        if framework == "scikitlearn":
            trainer = None

        return trainer

    return _get_adv_trainer


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 100
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.fixture()
def fix_get_cifar10_data():
    """
    Get the first 128 samples of the cifar10 test set

    :return: First 128 sample/label pairs of the cifar10 test dataset.
    """
    nb_test = 128

    (x_train, y_train), (x_test, y_test), _, _ = load_dataset("cifar10")
    y_test = np.argmax(y_test, axis=1)
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    x_test = np.transpose(x_test, (0, 3, 1, 2))  # return in channels first format

    y_train = np.argmax(y_train, axis=1)
    x_train, y_train = x_train[:nb_test], y_train[:nb_test]
    x_train = np.transpose(x_train, (0, 3, 1, 2))  # return in channels first format
    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test


@pytest.mark.only_with_platform("huggingface")
@pytest.mark.parametrize("label_format", ["one_hot", "numerical"])
def test_adversarial_trainer_trades_huggingface_fit_and_predict(
    get_adv_trainer, fix_get_cifar10_data, fix_get_mnist_subset, label_format
):

    import torch

    upsampler = torch.nn.Upsample(scale_factor=7, mode="nearest")
    if HF_MODEL_SIZE == "LARGE":
        (x_train, y_train, x_test, y_test) = fix_get_cifar10_data
        x_test_original = x_test.copy()

        x_test = np.float32(upsampler(torch.from_numpy(x_test)).cpu().numpy())
        x_train = np.float32(upsampler(torch.from_numpy(x_train)).cpu().numpy())
        x_test_original = np.float32(upsampler(torch.from_numpy(x_test_original)).cpu().numpy())
    else:
        (x_train, y_train, x_test, y_test) = fix_get_mnist_subset
        x_test_original = x_test.copy()

    if label_format == "one_hot":
        assert y_train.shape[-1] == 10
        assert y_test.shape[-1] == 10
    if label_format == "numerical":
        y_test = np.argmax(y_test, axis=1)
        y_train = np.argmax(y_train, axis=1)

    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return

    predictions = np.argmax(trainer.predict(x_test), axis=1)

    if label_format == "one_hot":
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / x_test.shape[0]
    else:
        accuracy = np.sum(predictions == y_test) / x_test.shape[0]

    if HF_MODEL_SIZE == "SMALL":
        nb_epochs = 20
    else:
        nb_epochs = 2

    trainer.fit(x_train, y_train, nb_epochs=nb_epochs)
    predictions_new = np.argmax(trainer.predict(x_test), axis=1)

    if label_format == "one_hot":
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / x_test.shape[0]
    else:
        accuracy_new = np.sum(predictions_new == y_test) / x_test.shape[0]

    np.testing.assert_array_almost_equal(
        float(np.mean(x_test_original - x_test)),
        0.0,
        decimal=4,
    )

    if HF_MODEL_SIZE == "SMALL":
        assert accuracy == 0.32
        assert accuracy_new > 0.32
    trainer.fit(x_train, y_train, nb_epochs=nb_epochs, validation_data=(x_train, y_train))


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("label_format", ["one_hot", "numerical"])
def test_adversarial_trainer_trades_pytorch_fit_and_predict(get_adv_trainer, fix_get_mnist_subset, label_format):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()

    if label_format == "one_hot":
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == "numerical":
        y_test_mnist = np.argmax(y_test_mnist, axis=1)
        y_train_mnist = np.argmax(y_train_mnist, axis=1)

    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return

    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)

    if label_format == "one_hot":
        accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy = np.sum(predictions == y_test_mnist) / x_test_mnist.shape[0]

    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)

    if label_format == "one_hot":
        accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy_new = np.sum(predictions_new == y_test_mnist) / x_test_mnist.shape[0]

    np.testing.assert_array_almost_equal(
        float(np.mean(x_test_mnist_original - x_test_mnist)),
        0.0,
        decimal=4,
    )
    print("accuracy ", accuracy)
    assert accuracy == 0.32
    assert accuracy_new > 0.32

    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("label_format", ["one_hot", "numerical"])
def test_adversarial_trainer_trades_pytorch_fit_generator_and_predict(
    get_adv_trainer, fix_get_mnist_subset, image_data_generator, label_format
):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()

    if label_format == "one_hot":
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == "numerical":
        y_test_mnist = np.argmax(y_test_mnist, axis=1)

    generator = image_data_generator()

    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return

    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)

    if label_format == "one_hot":
        accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy = np.sum(predictions == y_test_mnist) / x_test_mnist.shape[0]

    trainer.fit_generator(generator=generator, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)

    if label_format == "one_hot":
        accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy_new = np.sum(predictions_new == y_test_mnist) / x_test_mnist.shape[0]

    np.testing.assert_array_almost_equal(
        float(np.mean(x_test_mnist_original - x_test_mnist)),
        0.0,
        decimal=4,
    )

    assert accuracy == 0.32
    assert accuracy_new > 0.32
