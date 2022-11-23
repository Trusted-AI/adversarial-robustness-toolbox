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

import json
import os
import pytest

import numpy as np
import torch

from art.utils import load_dataset
from art.estimators.certification.deep_z import PytorchDeepZ
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

    return x_test, y_test


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
    return x_test, y_test


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_mnist_certification(art_warning, fix_get_mnist_data):
    """
    Check the following properties for the first 100 samples of the MNIST test set given an l_inft bound of 0.05.
        1) Upper and lower bounds are calculated correctly.
        2) The correct number of datapoints are certified.
        3) The standard accuracy is correct.
    """
    bound = 0.05

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)

    zonotope_model = PytorchDeepZ(
        model=ptc.model, clip_values=(0, 1), loss=torch.nn.CrossEntropyLoss(), input_shape=(1, 28, 28), nb_classes=10
    )

    correct_upper_bounds = np.asarray(
        json.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "certification/output_results/mini_nets/mnist_ub_results_0.05.json",
                )
            )
        )
    )
    correct_lower_bounds = np.asarray(
        json.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "certification/output_results/mini_nets/mnist_lb_results_0.05.json",
                )
            )
        )
    )

    num_cert = 0
    correct = 0
    try:
        for x, y in zip(fix_get_mnist_data[0], fix_get_mnist_data[1]):
            x = x.astype("float32")
            eps_bound = np.eye(784) * bound
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pred_sample = torch.from_numpy(pred_sample.astype("float32")).to(device)
            zonotope_model.model.set_forward_mode("concrete")
            prediction = zonotope_model.model.forward(pred_sample)
            prediction = np.argmax(prediction.cpu().detach().numpy())
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=x, eps=eps_bound)

            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            zonotope_model.model.set_forward_mode("abstract")

            bias, eps = zonotope_model.model.forward(eps=eps_bound, cent=data_sample_processed)

            upper_bounds, lower_bounds = zonotope_model.zonotope_get_bounds(bias, eps)
            if prediction == y:
                for bnds, correct_bds in zip(
                    [upper_bounds, lower_bounds], [correct_upper_bounds[correct], correct_lower_bounds[correct]]
                ):
                    bnds = torch.stack(bnds)
                    bnds = bnds.detach().cpu().numpy()
                    assert np.allclose(bnds, correct_bds, rtol=1e-05, atol=1e-05)

                correct += 1

            bias = bias.detach().cpu().numpy()
            eps = eps.detach().cpu().numpy()

            certified = True
            sub_certs = []
            min_bound_on_class = lower_bounds[y]

            for k in range(10):
                if k != prediction:
                    cert_via_sub = zonotope_model.certify_via_subtraction(
                        predicted_class=prediction, class_to_consider=k, cent=bias, eps=eps
                    )
                    sub_certs.append(cert_via_sub)

                    if min_bound_on_class <= upper_bounds[k]:
                        certified = False

            if certified:  # if box certified then subtract must also certify
                assert all(sub_certs)

            # make sure the certify method gives the same result
            is_certified = zonotope_model.certify(eps=eps_bound, cent=data_sample_processed, prediction=prediction)
            assert is_certified == all(sub_certs)

            if all(sub_certs) and int(prediction) == y:
                num_cert += 1

        assert num_cert == 94
        assert correct == 99

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_cifar_certification(art_warning, fix_get_cifar10_data):
    """
    Check the following properties for the first 100 samples of the CIFAR10 test set given an l_inft bound of 0.004.
        1) Upper and lower bounds are calculated correctly.
        2) The correct number of datapoints are certified.
        3) The standard accuracy is correct.
    """
    bound = 0.004

    ptc = get_cifar10_image_classifier_pt(from_logits=True)
    zonotope_model = PytorchDeepZ(
        model=ptc.model, clip_values=(0, 1), loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 32, 32), nb_classes=10
    )

    correct_upper_bounds = np.asarray(
        json.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "certification/output_results/mini_nets/cifar10_ub_results_0.004.json",
                )
            )
        )
    )
    correct_lower_bounds = np.asarray(
        json.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "certification/output_results/mini_nets/cifar10_lb_results_0.004.json",
                )
            )
        )
    )

    num_cert = 0
    correct = 0
    try:
        for x, y in zip(fix_get_cifar10_data[0], fix_get_cifar10_data[1]):
            x = x.astype("float32")
            eps_bound = np.eye(3 * 32 * 32) * bound
            x = np.moveaxis(x, [2], [0])
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pred_sample = torch.from_numpy(pred_sample.astype("float32")).to(device)
            zonotope_model.model.set_forward_mode("concrete")
            prediction = zonotope_model.model.forward(pred_sample)
            prediction = np.argmax(prediction.cpu().detach().numpy())
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=x, eps=eps_bound)

            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)
            zonotope_model.model.set_forward_mode("abstract")
            bias, eps_bound = zonotope_model.model.forward(eps=eps_bound, cent=data_sample_processed)

            upper_bounds, lower_bounds = zonotope_model.zonotope_get_bounds(bias, eps_bound)
            if prediction == y:
                for bnds, correct_bds in zip(
                    [upper_bounds, lower_bounds], [correct_upper_bounds[correct], correct_lower_bounds[correct]]
                ):
                    bnds = torch.stack(bnds)
                    bnds = bnds.detach().cpu().numpy()
                    assert np.allclose(bnds, correct_bds, rtol=1e-05, atol=1e-05)
                correct += 1

            bias = bias.detach().cpu().numpy()
            eps_bound = eps_bound.detach().cpu().numpy()

            certified = True
            sub_certs = []
            min_bound_on_class = lower_bounds[y]

            for k in range(10):
                if k != prediction:
                    cert_via_sub = zonotope_model.certify_via_subtraction(
                        predicted_class=prediction, class_to_consider=k, cent=bias, eps=eps_bound
                    )
                    sub_certs.append(cert_via_sub)

                    if min_bound_on_class <= upper_bounds[k]:
                        certified = False

            if certified:  # if box certified then subtract must also certify
                assert all(sub_certs)

            if all(sub_certs) and int(prediction) == y:
                num_cert += 1

        assert num_cert == 7
        assert correct == 8

    except ARTTestException as e:
        art_warning(e)
