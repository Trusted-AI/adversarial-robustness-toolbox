# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
import pytest

import numpy as np
import random
import torch

from art.utils import load_dataset
from art.estimators.certification.deep_z import PytorchDeepZ
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainerCertifiedPytorch

from tests.utils import ARTTestException
from tests.utils import get_image_classifier_pt, get_cifar10_image_classifier_pt


@pytest.fixture()
def fix_get_mnist_data():
    """
    Get the first 100 samples of the mnist test set with channels first format

    :return: First 100 sample/label pairs of the MNIST test dataset.
    """
    nb_test = 100

    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test = np.squeeze(x_test)
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_test, y_test = x_test[:nb_test], y_test[:nb_test]

    return x_test.astype(np.float32), y_test


@pytest.fixture()
def fix_get_cifar10_data():
    """
    Get the first 10 samples of the cifar10 test set

    :return: First 10 sample/label pairs of the cifar10 test dataset.
    """
    nb_test = 10

    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    y_test = np.argmax(y_test, axis=1)
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    return np.moveaxis(x_test, [3], [1]).astype(np.float32), y_test


@pytest.mark.skip_framework(
    "mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2", "tensorflow2v1", "huggingface"
)
def test_mnist_certified_training(art_warning, fix_get_mnist_data):
    """
    Check the following properties for the first 100 samples of the MNIST test set given an l_inft bound
        1) Check regular loss
        2) Train the model for 3 epochs using certified training.
        3) Re-Check regular loss
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    import torch.optim as optim

    optimizer = optim.Adam(ptc.model.parameters(), lr=1e-4)
    zonotope_model = PytorchDeepZ(
        model=ptc.model.to(device),
        optimizer=optimizer,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    pgd_params = {"eps": 0.25, "eps_step": 0.05, "max_iter": 20, "num_random_init": 0, "batch_size": 64}

    trainer = AdversarialTrainerCertifiedPytorch(zonotope_model, pgd_params=pgd_params, batch_size=10, bound=0.1)

    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    try:
        # Check losses pre-training
        trainer._classifier.model.set_forward_mode("concrete")
        prediction = trainer._classifier.model.forward(fix_get_mnist_data[0])
        loss = trainer._classifier.concrete_loss(prediction, torch.tensor(fix_get_mnist_data[1]).to(device))
        assert round(float(loss.cpu().detach().numpy()), 3) == 0.095

        trainer.fit(fix_get_mnist_data[0], fix_get_mnist_data[1], nb_epochs=3)

        # Check losses post-training
        prediction = trainer._classifier.model.forward(fix_get_mnist_data[0])
        loss = trainer._classifier.concrete_loss(prediction, torch.tensor(fix_get_mnist_data[1]).to(device))
        assert round(float(loss.cpu().detach().numpy()), 3) == 0.092

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2", "tensorflow2v1", "huggingface"
)
def test_mnist_certified_loss(art_warning, fix_get_mnist_data):
    """
    Check the certified losses with interval_loss_cce, max_logit_loss, and make sure that we give a lower
    bound compared to PGD.
    """
    bound = 0.05
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    import torch.optim as optim

    optimizer = optim.Adam(ptc.model.parameters(), lr=1e-4)
    zonotope_model = PytorchDeepZ(
        model=ptc.model.to(device),
        optimizer=optimizer,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    pgd_params = {"eps": 0.25, "eps_step": 0.05, "max_iter": 20, "num_random_init": 0, "batch_size": 64}

    trainer = AdversarialTrainerCertifiedPytorch(zonotope_model, pgd_params=pgd_params, batch_size=10, bound=bound)

    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    concrete_correct = 0
    samples_x, samples_eps = trainer.predict_zonotopes(np.copy(fix_get_mnist_data[0]), bound)
    try:
        certified_loss = 0
        for i, (x, y) in enumerate(zip(fix_get_mnist_data[0], fix_get_mnist_data[1])):
            x = x.astype("float32")
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            zonotope_model.model.set_forward_mode("concrete")
            prediction = trainer._classifier.model.forward(torch.from_numpy(pred_sample.astype("float32")).to(device))
            if np.argmax(prediction.cpu().detach().numpy()) == y:
                concrete_correct += 1

            eps_bound = np.eye(784) * bound
            zonotope_model.model.set_forward_mode("abstract")
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=x, eps=eps_bound)
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            bias, eps_bound = trainer._classifier.model.forward(eps=eps_bound, cent=data_sample_processed)
            bias = torch.unsqueeze(bias, dim=0)

            assert np.allclose(samples_x[i], bias.cpu().detach().numpy())
            assert np.allclose(samples_eps[i], eps_bound.cpu().detach().numpy())

            certified_loss += trainer._classifier.interval_loss_cce(
                prediction=torch.cat((bias, eps_bound)),
                target=torch.from_numpy(np.expand_dims(y, axis=0)).to(device),
            )

        assert round(float(certified_loss.cpu().detach().numpy()), 4) == 39.7759

        samples_certified = 0

        certified_loss = 0
        for x, y in zip(fix_get_mnist_data[0], fix_get_mnist_data[1]):
            x = x.astype("float32")
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            zonotope_model.model.set_forward_mode("concrete")
            prediction = trainer._classifier.model.forward(torch.from_numpy(pred_sample.astype("float32")).to(device))
            if np.argmax(prediction.cpu().detach().numpy()) == y:
                concrete_correct += 1

            eps_bound = np.eye(784) * bound
            zonotope_model.model.set_forward_mode("abstract")
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=np.copy(x), eps=eps_bound)
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            bias, eps_bound = trainer._classifier.model.forward(eps=eps_bound, cent=data_sample_processed)
            bias = torch.unsqueeze(bias, dim=0)

            sample_loss = trainer._classifier.max_logit_loss(
                prediction=torch.cat((bias, eps_bound)),
                target=torch.from_numpy(np.expand_dims(y, axis=0)).to(device),
            )

            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=np.copy(x), eps=np.eye(784) * bound)
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            is_certified = zonotope_model.certify(
                eps=eps_bound, cent=data_sample_processed, prediction=np.argmax(prediction.cpu().detach().numpy())
            )

            if (
                is_certified and int(np.argmax(prediction.cpu().detach().numpy())) == y
            ):  # if it is classified correctly AND is certified
                assert sample_loss <= 0.0  # ...the max_logit_loss gives negatve.
                samples_certified += 1

            certified_loss += sample_loss

        assert float(certified_loss.cpu().detach().numpy()) == pytest.approx(-309.2724, abs=0.001)
        assert samples_certified == 94

        # empirically check that PGD does not give a lower acc
        zonotope_model.model.set_forward_mode("concrete")
        attack = ProjectedGradientDescent(
            estimator=trainer._classifier,
            eps=bound,
            eps_step=bound / 20,
            max_iter=30,
            num_random_init=0,
        )
        i_batch = attack.generate(fix_get_mnist_data[0], y=fix_get_mnist_data[1])
        preds = trainer._classifier.model.forward(i_batch)
        preds = preds.cpu().detach().numpy()
        acc = np.sum(np.argmax(preds, axis=1) == fix_get_mnist_data[1]) / len(fix_get_mnist_data[1])
        assert acc * 100 >= samples_certified

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2", "tensorflow2v1", "huggingface"
)
def test_cifar_certified_training(art_warning, fix_get_cifar10_data):
    """
    Check the following properties for the first 10 samples of the CIFAR test set given an l_inft bound
        1) Check regular loss
        2) Train the model for 3 epochs using certified training.
        3) Re-Check regular loss
    """

    bound = 0.01

    ptc = get_cifar10_image_classifier_pt(from_logits=True)

    import torch.optim as optim

    optimizer = optim.Adam(ptc.model.parameters(), lr=1e-4)
    zonotope_model = PytorchDeepZ(
        model=ptc.model,
        optimizer=optimizer,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    pgd_params = {"eps": 8 / 255, "eps_step": 1 / 255, "max_iter": 20, "num_random_init": 0, "batch_size": 64}

    trainer = AdversarialTrainerCertifiedPytorch(zonotope_model, pgd_params=pgd_params, batch_size=10, bound=bound)

    np.random.seed(123)
    random.seed(123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        trainer._classifier.model.set_forward_mode("concrete")
        prediction = trainer._classifier.model.forward(fix_get_cifar10_data[0])
        loss = trainer._classifier.concrete_loss(prediction, torch.tensor(fix_get_cifar10_data[1]).to(device))
        assert round(float(loss.cpu().detach().numpy()), 4) == 1.0611

        trainer.fit(fix_get_cifar10_data[0], fix_get_cifar10_data[1], nb_epochs=3)
        prediction = trainer._classifier.model.forward(fix_get_cifar10_data[0])

        loss = trainer._classifier.concrete_loss(prediction, torch.tensor(fix_get_cifar10_data[1]).to(device))
        assert round(float(loss.cpu().detach().numpy()), 4) == 1.0092
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2", "tensorflow2v1", "huggingface"
)
def test_cifar_certified_loss(art_warning, fix_get_cifar10_data):
    """
    Check the certified losses with interval_loss_cce, max_logit_loss, and make sure that we give a lower
    bound compared to PGD.
    """
    bound = 2 / 255
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ptc = get_cifar10_image_classifier_pt(from_logits=True)

    import torch.optim as optim

    optimizer = optim.Adam(ptc.model.parameters(), lr=1e-4)
    zonotope_model = PytorchDeepZ(
        model=ptc.model,
        optimizer=optimizer,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    pgd_params = {"eps": 8 / 255, "eps_step": 1 / 255, "max_iter": 20, "num_random_init": 0, "batch_size": 64}

    trainer = AdversarialTrainerCertifiedPytorch(zonotope_model, pgd_params=pgd_params, batch_size=10, bound=bound)

    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    concrete_correct = 0
    samples_x, samples_eps = trainer.predict_zonotopes(np.copy(fix_get_cifar10_data[0]), bound)
    try:
        certified_loss = 0
        for x, y in zip(fix_get_cifar10_data[0], fix_get_cifar10_data[1]):
            x = x.astype("float32")
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            zonotope_model.model.set_forward_mode("concrete")
            prediction = trainer._classifier.model.forward(torch.from_numpy(pred_sample.astype("float32")).to(device))
            if np.argmax(prediction.cpu().detach().numpy()) == y:
                concrete_correct += 1

            eps_bound = np.eye(3 * 32 * 32) * bound
            zonotope_model.model.set_forward_mode("abstract")
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=x, eps=eps_bound)
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            bias, eps_bound = trainer._classifier.model.forward(eps=eps_bound, cent=data_sample_processed)
            bias = torch.unsqueeze(bias, dim=0)

            certified_loss += trainer._classifier.interval_loss_cce(
                prediction=torch.cat((bias, eps_bound)),
                target=torch.from_numpy(np.expand_dims(y, axis=0)).to(device),
            )

        certified_loss = certified_loss.cpu().detach().numpy()
        assert round(float(certified_loss), 4) == 16.5492

        samples_certified = 0

        certified_loss = 0
        for i, (x, y) in enumerate(zip(fix_get_cifar10_data[0], fix_get_cifar10_data[1])):
            x = x.astype("float32")
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            zonotope_model.model.set_forward_mode("concrete")
            prediction = trainer._classifier.model.forward(torch.from_numpy(pred_sample.astype("float32")).to(device))
            if np.argmax(prediction.cpu().detach().numpy()) == y:
                concrete_correct += 1

            eps_bound = np.eye(3 * 32 * 32) * bound
            zonotope_model.model.set_forward_mode("abstract")
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=np.copy(x), eps=eps_bound)
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            bias, eps_bound = trainer._classifier.model.forward(eps=eps_bound, cent=data_sample_processed)
            bias = torch.unsqueeze(bias, dim=0)

            assert np.allclose(samples_x[i], bias.cpu().detach().numpy())
            assert np.allclose(samples_eps[i], eps_bound.cpu().detach().numpy())

            sample_loss = trainer._classifier.max_logit_loss(
                prediction=torch.cat((bias, eps_bound)),
                target=torch.from_numpy(np.expand_dims(y, axis=0)).to(device),
            )

            data_sample_processed, eps_bound = zonotope_model.pre_process(
                cent=np.copy(x), eps=np.eye(3 * 32 * 32) * bound
            )
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            is_certified = zonotope_model.certify(
                eps=eps_bound, cent=data_sample_processed, prediction=np.argmax(prediction.cpu().detach().numpy())
            )

            if (
                is_certified and int(np.argmax(prediction.cpu().detach().numpy())) == y
            ):  # if it is classified correctly AND is certified
                assert sample_loss <= 0.0  # ...the max_logit_loss gives negatve.
                samples_certified += 1

            certified_loss += sample_loss

        assert round(float(certified_loss.cpu().detach().numpy()), 4) == 1.6515
        assert samples_certified == 6

        # empirically check that PGD does not give a lower acc
        zonotope_model.model.set_forward_mode("concrete")
        attack = ProjectedGradientDescent(
            estimator=trainer._classifier,
            eps=bound,
            eps_step=bound / 20,
            max_iter=30,
            num_random_init=0,
        )
        i_batch = attack.generate(fix_get_cifar10_data[0], y=fix_get_cifar10_data[1])
        preds = trainer._classifier.model.forward(i_batch)
        preds = preds.cpu().detach().numpy()
        acc = np.sum(np.argmax(preds, axis=1) == fix_get_cifar10_data[1]) / len(fix_get_cifar10_data[1])
        assert acc * 100 >= samples_certified

    except ARTTestException as e:
        art_warning(e)
