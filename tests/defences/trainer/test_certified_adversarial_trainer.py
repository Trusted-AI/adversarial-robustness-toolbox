import json
import os
import pytest

import numpy as np
import torch

from art.utils import load_dataset
from art.estimators.certification.deep_z import PytorchDeepZ
from tests.utils import ARTTestException
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from tests.utils import get_image_classifier_pt, get_cifar10_image_classifier_pt

from art.defences.trainer import AdversarialTrainerCertified

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


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_mnist_certified_training(art_warning, fix_get_mnist_data):
    """
    Check the following properties for the first 100 samples of the MNIST test set given an l_inft bound of 0.05.
        1) Upper and lower bounds are calculated correctly.
        2) The correct number of datapoints are certified.
        3) The standard accuracy is correct.
    """
    bound = 0.05

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    import torch.optim as optim

    optimizer = optim.Adam(ptc.model.parameters(), lr=1e-4)
    zonotope_model = PytorchDeepZ(
        model=ptc.model,
        optimizer=optimizer,
        clip_values=(0, 1), loss=torch.nn.CrossEntropyLoss(), input_shape=(1, 28, 28), nb_classes=10
    )

    pgd_params = {"eps": 0.25,
                  "eps_step": 0.05,
                  "max_iter": 20,
                  "num_random_init": 0,
                  "batch_size": 64}

    trainer = AdversarialTrainerCertified(zonotope_model,
                                          pgd_params=pgd_params,
                                          batch_size=10,
                                          bound=0.1)

    import random
    np.random.seed(123)
    random.seed(123)

    trainer.fit(fix_get_mnist_data[0],
                fix_get_mnist_data[1],
                nb_epochs=3)

    concrete_correct = 0

    # check adversarial example performance
    trainer.set_forward_mode("concrete")
    attack = ProjectedGradientDescent(
        estimator=trainer._classifier,
        eps=pgd_params["eps"],
        eps_step=pgd_params["eps_step"],
        max_iter=pgd_params["max_iter"],
        num_random_init=pgd_params["num_random_init"],
    )

    try:
        i_batch = attack.generate(fix_get_mnist_data[0], y=fix_get_mnist_data[1])
        prediction = trainer.predict(i_batch)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loss = trainer._classifier.concrete_loss(prediction,
                                                 torch.tensor(fix_get_mnist_data[1]).to(device))

        loss = loss.cpu().detach().numpy()
        assert round(float(loss), 5) == 8.65328
    except ARTTestException as e:
        art_warning(e)

    try:
        certified_loss = 0
        for x, y in zip(fix_get_mnist_data[0], fix_get_mnist_data[1]):
            x = x.astype("float32")
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            zonotope_model.set_forward_mode("concrete")
            prediction = trainer.predict(torch.from_numpy(pred_sample.astype("float32")).to(device))
            if np.argmax(prediction.cpu().detach().numpy()) == y:
                concrete_correct += 1

            eps_bound = np.eye(784) * bound
            zonotope_model.set_forward_mode("abstract")
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=x, eps=eps_bound)
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            bias, eps_bound = trainer._classifier.forward(eps=eps_bound, cent=data_sample_processed)
            bias = torch.unsqueeze(bias, dim=0)

            certified_loss += trainer._classifier.interval_loss_cce(
                prediction=torch.cat((bias, eps_bound)),
                target=torch.from_numpy(np.expand_dims(y, axis=0)).to(device),
            )

        certified_loss = certified_loss.cpu().detach().numpy()
        assert round(float(certified_loss), 4) == 38.9669
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_cifar_certified_training(art_warning, fix_get_cifar10_data):
    """
    Check the following properties for the first 10 samples of the CIFAR10 test set given an l_inft bound of 0.004.
        1) Upper and lower bounds are calculated correctly.
        2) The correct number of datapoints are certified.
        3) The standard accuracy is correct.
    """

    bound = 0.01

    ptc = get_cifar10_image_classifier_pt(from_logits=True)

    import torch.optim as optim
    optimizer = optim.Adam(ptc.model.parameters(), lr=1e-4)
    zonotope_model = PytorchDeepZ(
        model=ptc.model,
        optimizer=optimizer,
        clip_values=(0, 1), loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 32, 32), nb_classes=10
    )

    pgd_params = {"eps": 8 / 255,
                  "eps_step": 1/ 255,
                  "max_iter": 20,
                  "num_random_init": 0,
                  "batch_size": 64}

    trainer = AdversarialTrainerCertified(zonotope_model,
                                          pgd_params=pgd_params,
                                          batch_size=10,
                                          bound=bound)

    import random
    np.random.seed(123)
    random.seed(123)

    trainer.fit(fix_get_cifar10_data[0],
                fix_get_cifar10_data[1],
                nb_epochs=3)

    concrete_correct = 0

    # check adversarial example performance
    attack = ProjectedGradientDescent(
        estimator=trainer._classifier,
        eps=pgd_params["eps"],
        eps_step=pgd_params["eps_step"],
        max_iter=pgd_params["max_iter"],
        num_random_init=pgd_params["num_random_init"],
    )

    i_batch = attack.generate(fix_get_cifar10_data[0], y=fix_get_cifar10_data[1])
    prediction = trainer.predict(i_batch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss = trainer._classifier.concrete_loss(prediction,
                                             torch.tensor(fix_get_cifar10_data[1]).to(device))

    loss = loss.cpu().detach().numpy()
    assert round(float(loss), 5) == 2.14333

    try:
        certified_loss = 0
        for x, y in zip(fix_get_cifar10_data[0], fix_get_cifar10_data[1]):
            x = x.astype("float32")
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            zonotope_model.set_forward_mode("concrete")
            prediction = trainer.predict(torch.from_numpy(pred_sample.astype("float32")).to(device))
            if np.argmax(prediction.cpu().detach().numpy()) == y:
                concrete_correct += 1

            eps_bound = np.eye(3 * 32 * 32) * bound
            zonotope_model.set_forward_mode("abstract")
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=x, eps=eps_bound)
            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            bias, eps_bound = trainer._classifier.forward(eps=eps_bound, cent=data_sample_processed)
            bias = torch.unsqueeze(bias, dim=0)

            certified_loss += trainer._classifier.interval_loss_cce(
                prediction=torch.cat((bias, eps_bound)),
                target=torch.from_numpy(np.expand_dims(y, axis=0)).to(device),
            )

        certified_loss = certified_loss.cpu().detach().numpy()
        assert round(float(certified_loss), 4) == 17.9115
    except ARTTestException as e:
        art_warning(e)