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

from art.defences.trainer import AdversarialTrainerOAATPyTorch
from art.attacks.evasion import ProjectedGradientDescent


@pytest.fixture()
def get_adv_trainer_oaat(framework, image_dl_estimator):
    def _get_adv_trainer_oaat():

        if framework == "keras":
            trainer = None
        if framework in ["tensorflow", "tensorflow2v1"]:
            trainer = None
        if framework == "pytorch":
            classifier, _ = image_dl_estimator(from_logits=True)
            proxy_classifier, _ = image_dl_estimator(from_logits=True)
            lpips_classifier, _ = image_dl_estimator(from_logits=True)
            list_avg_classifiers = [image_dl_estimator(from_logits=True)[0] for _ in range(3)]
            train_params = {
                "weight_decay": 0.0,
                "lr": 0.1,
                "momentum": 0.0,
                "norm": np.inf,
                "i_epsilon": 0.1,
                "epsilon": 0.2,
                "alpha": 1.0,
                "beta": 1.5,
                "awp_gamma": 0.005,
                "awp_warmup": 1,
                "lr_schedule": "cosine",
                "beta_final": 3.0,
                "mixup_alpha": 0.2,
                "mixup_epsilon": 0.25,
                "lpips_weight": 1.0,
                "list_swa_tau": [0.995, 0.9996, 0.9998],
                "oaat_warmup": 0,
                "alternate_iter_eps": 0.15,
                "swa_save_epoch": 0,
                "list_swa_epoch": [0, 15, 15],
                "max_iter": 20,
                "models_path": None,
                "load_swa_model_tau": 0.995,
                "layer_names_activation": ["conv"],
            }

            attack = ProjectedGradientDescent(
                classifier,
                norm=np.inf,
                eps=0.2,
                eps_step=0.02,
                max_iter=20,
                targeted=False,
                num_random_init=1,
                batch_size=16,
                verbose=False,
            )
            trainer = AdversarialTrainerOAATPyTorch(
                classifier, proxy_classifier, lpips_classifier, list_avg_classifiers, attack, train_params
            )
        if framework == "scikitlearn":
            trainer = None

        return trainer

    return _get_adv_trainer_oaat


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 100
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("label_format", ["one_hot", "numerical"])
def test_adversarial_trainer_oaat_pytorch_fit_and_predict(get_adv_trainer_oaat, fix_get_mnist_subset, label_format):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()

    if label_format == "one_hot":
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == "numerical":
        y_train_mnist = np.argmax(y_train_mnist, axis=1)
        y_test_mnist = np.argmax(y_test_mnist, axis=1)

    trainer = get_adv_trainer_oaat()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return

    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)

    if label_format == "one_hot":
        accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy = np.sum(predictions == y_test_mnist) / x_test_mnist.shape[0]

    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20, batch_size=16)
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

    trainer.fit(
        x_train_mnist, y_train_mnist, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist), batch_size=16
    )


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("label_format", ["one_hot", "numerical"])
def test_adversarial_trainer_oaat_pytorch_fit_generator_and_predict(
    get_adv_trainer_oaat, fix_get_mnist_subset, image_data_generator, label_format
):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()

    if label_format == "one_hot":
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == "numerical":
        y_test_mnist = np.argmax(y_test_mnist, axis=1)

    generator = image_data_generator()

    trainer = get_adv_trainer_oaat()
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

    trainer.fit_generator(generator=generator, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))
