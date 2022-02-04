
from art.utils import load_dataset
from art.estimators.certification.deep_z import PytorchDeepZ
import pytest
import numpy as np
import json
from torch import nn
import torch
from tests.utils import ARTTestException
from tests.utils import (
    master_seed,
    get_image_classifier_pt,
    get_cifar10_image_classifier_pt)

NB_TEST = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def fix_get_mnist_data():
    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test = np.squeeze(x_test)
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

    return x_test, y_test


@pytest.fixture()
def fix_get_cifar10_data():
    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    y_test = np.argmax(y_test, axis=1)
    x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
    return x_test, y_test


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2v1")
def test_mnist_certification(art_warning, fix_get_mnist_data, image_dl_estimator):
    bound = 0.05

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)

    zonotope_model = PytorchDeepZ(model=ptc.model,
                                  clip_values=(0, 1),
                                  loss=nn.CrossEntropyLoss(),
                                  input_shape=(1, 28, 28),
                                  nb_classes=10,
                                  device_type=device)

    f = open('output_results/mini_nets/mnist_ub_results_0.05.json')
    correct_upper_bounds = np.asarray(json.load(f))
    f = open('output_results/mini_nets/mnist_lb_results_0.05.json')
    correct_lower_bounds = np.asarray(json.load(f))

    num_cert = 0
    correct = 0
    try:
        for x, y in zip(fix_get_mnist_data[0], fix_get_mnist_data[1]):
            x = x.astype('float32')
            eps_bound = np.eye(784) * bound
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            prediction = zonotope_model.predict(pred_sample)
            prediction = np.argmax(prediction)
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=np.copy(x),
                                                                          eps=eps_bound)

            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)

            bias, eps_bound = zonotope_model.forward(eps=eps_bound,
                                                     cent=data_sample_processed)

            upper_bounds, lower_bounds = zonotope_model.zonotope_get_bounds(bias, eps_bound)
            if prediction == y:
                for bnds, correct_bds in zip([upper_bounds, lower_bounds], [correct_upper_bounds[correct], correct_lower_bounds[correct]]):
                    bnds = torch.stack(bnds)
                    bnds = bnds.detach().cpu().numpy()
                    assert np.allclose(bnds, correct_bds, rtol=1e-06, atol=1e-06)

                correct += 1

            bias = bias.detach().cpu().numpy()
            eps_bound = eps_bound.detach().cpu().numpy()

            certified = True
            sub_certs = []
            min_bound_on_class = lower_bounds[y]

            for k in range(10):
                if k != prediction:
                    cert_via_sub = zonotope_model.certify_via_subtraction(predicted_class=prediction,
                                                                          class_to_consider=k,
                                                                          cent=bias,
                                                                          eps=eps_bound)
                    sub_certs.append(cert_via_sub)

                    if min_bound_on_class <= upper_bounds[k]:
                        certified = False

            if certified:  # if box certified then subtract must also certify
                assert all(sub_certs)

            if all(sub_certs) and int(prediction) == y:
                num_cert += 1

        assert num_cert == 94

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2v1")
def test_cifar_certification(art_warning, fix_get_cifar10_data, image_dl_estimator):
    bound = 0.004

    ptc = get_cifar10_image_classifier_pt(from_logits=True)
    zonotope_model = PytorchDeepZ(model=ptc.model,
                                  clip_values=(0, 1),
                                  loss=nn.CrossEntropyLoss(),
                                  input_shape=(3, 32, 32),
                                  nb_classes=10,
                                  device_type=device)

    f = open('output_results/mini_nets/cifar10_ub_results_0.004.json')
    correct_upper_bounds = np.asarray(json.load(f))
    f = open('output_results/mini_nets/cifar10_lb_results_0.004.json')
    correct_lower_bounds = np.asarray(json.load(f))

    num_cert = 0
    correct = 0
    try:
        for x, y in zip(fix_get_cifar10_data[0], fix_get_cifar10_data[1]):
            x = x.astype('float32')
            eps_bound = np.eye(3*32*32) * bound
            x = np.moveaxis(x, [2], [0])
            pred_sample = np.copy(x)
            pred_sample = np.expand_dims(pred_sample, axis=0)
            prediction = zonotope_model.predict(pred_sample)
            prediction = np.argmax(prediction)
            data_sample_processed, eps_bound = zonotope_model.pre_process(cent=np.copy(x),
                                                                          eps=eps_bound)

            data_sample_processed = np.expand_dims(data_sample_processed, axis=0)

            bias, eps_bound = zonotope_model.forward(eps=eps_bound,
                                                     cent=data_sample_processed)

            upper_bounds, lower_bounds = zonotope_model.zonotope_get_bounds(bias, eps_bound)
            if prediction == y:
                for bnds, correct_bds in zip([upper_bounds, lower_bounds], [correct_upper_bounds[correct], correct_lower_bounds[correct]]):
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
                    cert_via_sub = zonotope_model.certify_via_subtraction(predicted_class=prediction,
                                                                          class_to_consider=k,
                                                                          cent=bias,
                                                                          eps=eps_bound)
                    sub_certs.append(cert_via_sub)

                    if min_bound_on_class <= upper_bounds[k]:
                        certified = False

            if certified:  # if box certified then subtract must also certify
                assert all(sub_certs)

            if all(sub_certs) and int(prediction) == y:
                num_cert += 1

        assert num_cert == 52
        assert correct == 61

    except ARTTestException as e:
        art_warning(e)

