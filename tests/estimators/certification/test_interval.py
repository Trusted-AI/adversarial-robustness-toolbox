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

import torch
import numpy as np

from art.estimators.certification.interval import PyTorchIBPClassifier
from art.estimators.certification.interval import PyTorchIntervalConv2D

from art.utils import load_dataset
from tests.utils import ARTTestException
from tests.utils import get_image_classifier_pt


def get_synthetic_model():
    """
    Get a model with just one convolutional layer to test the convolutional to dense conversion
    """

    class SyntheticIntervalModel(torch.nn.Module):
        def __init__(
            self, input_shape, output_channels, kernel_size, stride=1, bias=False, padding=0, dilation=1, to_debug=True
        ):
            super().__init__()

            self.conv1 = PyTorchIntervalConv2D(
                in_channels=input_shape[1],
                out_channels=output_channels,
                kernel_size=kernel_size,
                input_shape=input_shape,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                to_debug=to_debug,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )

        def forward(self, x):
            """
            Computes the forward pass though the neural network
            :param x: input data of the form [number of samples, interval, feature]
            :return:
            """
            return self.conv1.concrete_forward(x)

    return SyntheticIntervalModel


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


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_single_channel_in_multi_out(art_warning):
    """
    Check that the conversion works for a single input channel.
    """
    synth_model = get_synthetic_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 1, 25, 25).to(device)
    model = synth_model(input_shape=synthetic_data.shape, output_channels=4, kernel_size=5)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv_debug(synthetic_data)
    output_from_equivalent = torch.reshape(output_from_equivalent, output_from_conv.shape)
    try:
        assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_multi_channel_in_single_out():
    """
    Check that the conversion works for multiple input channels with single output
    """
    synth_model = get_synthetic_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = synth_model(input_shape=synthetic_data.shape, output_channels=1, kernel_size=5)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv_debug(synthetic_data)

    output_from_equivalent = output_from_equivalent.flatten()
    output_from_conv = output_from_conv.flatten()

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_multi_channel_in_multi_out():
    """
    Check that the conversion works for multiple input channels and multiple output channels.
    """
    synth_model = get_synthetic_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = synth_model(input_shape=synthetic_data.shape, output_channels=12, kernel_size=5)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv_debug(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_layer_multi_channel_in_multi_out_with_stride():
    """
    Check that the conversion works for multiple input/output channels with strided convolution
    """
    synth_model = get_synthetic_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = synth_model(input_shape=synthetic_data.shape, output_channels=12, kernel_size=5, stride=2)

    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv_debug(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_layer_multi_channel_in_multi_out_with_stride_and_bias():
    """
    Check that the conversion works for multiple input/output channels with strided convolution and bias
    """
    synth_model = get_synthetic_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = synth_model(input_shape=synthetic_data.shape, output_channels=12, kernel_size=5, bias=True, stride=2)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv_debug(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_layer_padding():
    """
    Check that the conversion works for multiple input/output channels with strided convolution, bias,
    and padding
    """
    synth_model = get_synthetic_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = synth_model(
        input_shape=synthetic_data.shape, output_channels=12, kernel_size=5, bias=True, padding=2, stride=2
    )
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv_debug(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_layer_dilation():
    """
    Check that the conversion works for multiple input/output channels with strided convolution, bias,
    padding, and dilation
    """
    synth_model = get_synthetic_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = synth_model(
        input_shape=synthetic_data.shape, output_channels=12, kernel_size=5, bias=True, padding=2, stride=2, dilation=3
    )
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv_debug(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_layer_grads():
    """
    Checking that the gradients are correctly backpropagated through the convolutional layer
    """
    synth_model = get_synthetic_model()
    output_channels = 12
    input_channels = 3
    loss_fn = torch.nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.rand(size=(32, 12, 21, 21)).to(device)

    synthetic_data = torch.rand(32, input_channels, 25, 25).to(device)
    model = synth_model(
        input_shape=synthetic_data.shape, output_channels=output_channels, kernel_size=5, bias=True, stride=1
    )
    model = model.to(device)
    output_from_equivalent = model.forward(synthetic_data)
    loss = loss_fn(output_from_equivalent, target)
    loss.backward()

    equivalent_grads = model.conv1.conv.weight.grad
    equivalent_grads = (
        torch.reshape(equivalent_grads, shape=(output_channels, input_channels, 5, 5)).detach().clone().to(device)
    )
    equivalent_bias = model.conv1.bias_to_grad.grad.data.detach().clone().to(device)

    model.zero_grad()
    output_from_conv = model.conv1.conv_debug(synthetic_data)
    loss = loss_fn(output_from_conv, target)
    loss.backward()

    assert torch.allclose(equivalent_grads, model.conv1.conv_debug.weight.grad, atol=1e-05)
    assert torch.allclose(equivalent_bias, model.conv1.conv_debug.bias.grad, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_train_loop():
    """
    Assert that training the interval conv layer gives the same results as a regular layer
    """
    output_channels = 12
    input_channels = 3
    loss_fn = torch.nn.MSELoss()
    synth_model = get_synthetic_model()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.rand(size=(32, 12, 21, 21)).to(device)

    synthetic_data = torch.rand(32, input_channels, 25, 25).to(device)
    model = synth_model(
        input_shape=synthetic_data.shape,
        output_channels=output_channels,
        kernel_size=5,
        bias=True,
        stride=1,
        to_debug=False,
    ).to(device)

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=12,
                kernel_size=5,
                bias=True,
                stride=1,
            )

        def forward(self, x):
            return self.conv(x)

    test_model = TestModel()
    # Get the weights we will transfer over
    # Set the weights in the normal model
    test_model.conv.weight = torch.nn.Parameter(
        torch.reshape(
            torch.tensor(model.conv1.conv.weight.data.cpu().detach().numpy()),
            shape=(output_channels, input_channels, 5, 5),
        )
    )
    test_model.conv.bias = torch.nn.Parameter(torch.tensor(model.conv1.bias_to_grad.data.cpu().detach().numpy()))
    test_model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    test_opt = torch.optim.Adam(test_model.parameters(), lr=0.0001)

    for _ in range(5):
        model.zero_grad()

        # Test model is for ground truth comparison
        test_model.zero_grad()
        output_from_equivalent = model.forward(synthetic_data)
        loss = loss_fn(output_from_equivalent, target)
        loss.backward()

        equivalent_grads = model.conv1.conv.weight.grad
        equivalent_grads = (
            torch.reshape(equivalent_grads, shape=(output_channels, input_channels, 5, 5)).detach().clone().to(device)
        )
        equivalent_bias = model.conv1.bias_to_grad.grad.clone().to(device)

        output_from_conv = test_model.forward(synthetic_data)
        loss = loss_fn(output_from_conv, target)
        loss.backward()

        assert torch.allclose(equivalent_grads, test_model.conv.weight.grad, atol=1e-05)
        assert torch.allclose(equivalent_bias, test_model.conv.bias.grad, atol=1e-05)

        optimizer.step()
        test_opt.step()

        reshaped_weights = torch.reshape(
            model.conv1.conv.weight.data.clone().detach(), shape=(output_channels, input_channels, 5, 5)
        ).to(device)

        assert torch.allclose(reshaped_weights, test_model.conv.weight.data, atol=1e-05)

        # function to re-pop the dense layer
        model.conv1.re_convert(device)

        # Sanity check! Are the grads still present and the same after the conversion?
        equivalent_grads = model.conv1.conv.weight.grad
        equivalent_grads = (
            torch.reshape(equivalent_grads, shape=(output_channels, input_channels, 5, 5)).detach().clone().to(device)
        )
        equivalent_bias = model.conv1.bias_to_grad.grad

        assert torch.allclose(equivalent_grads, test_model.conv.weight.grad, atol=1e-05)
        assert torch.allclose(equivalent_bias, test_model.conv.bias.grad, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_mnist_certification(art_warning, fix_get_mnist_data):
    """
    Assert the certification performance on sample MNIST data
    """
    bound = 0.05
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)

    box_model = PyTorchIBPClassifier(
        model=ptc.model.to(device),
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    mnist_data = fix_get_mnist_data[0]
    mnist_labels = fix_get_mnist_data[1]
    box_model.model.set_forward_mode("concrete")
    preds = box_model.predict(mnist_data.astype("float32"))
    acc = np.sum(np.argmax(preds, axis=1) == mnist_labels)
    assert acc == 99

    box_model.model.set_forward_mode("abstract")
    up_bound = np.expand_dims(np.clip(mnist_data + bound, 0, 1), axis=1)
    low_bound = np.expand_dims(np.clip(mnist_data - bound, 0, 1), axis=1)
    interval_x = np.concatenate([low_bound, up_bound], axis=1)

    interval_preds = box_model.predict_intervals(interval_x, is_interval=True)
    cert_results = box_model.certify(preds=interval_preds, labels=mnist_labels)
    assert np.sum(cert_results) == 48


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_mnist_certification_conversion(art_warning, fix_get_mnist_data):
    """
    Assert that the re-convert method does not throw errors
    """

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.conv1 = torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                bias=True,
                stride=1,
            ).to(self.device)

            self.conv2 = torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                bias=True,
                stride=1,
            ).to(self.device)

            self.fc1 = torch.nn.Linear(in_features=12800, out_features=100)
            self.fc2 = torch.nn.Linear(in_features=100, out_features=10)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.reshape(x.shape[0], -1)
            x = self.relu(self.fc1(x))

            return self.fc2(x)

    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TestModel().to(device)

    box_model = PyTorchIBPClassifier(
        model=model, clip_values=(0, 1), loss=torch.nn.CrossEntropyLoss(), input_shape=(1, 28, 28), nb_classes=10
    )

    mnist_data = fix_get_mnist_data[0]
    mnist_labels = torch.tensor(fix_get_mnist_data[1]).to(device)
    box_model.model.set_forward_mode("concrete")
    for _ in range(5):
        preds = box_model.model.forward(mnist_data)
        loss = loss_fn(preds, mnist_labels)
        loss.backward()
        box_model.model.zero_grad()
        box_model.model.re_convert()


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_certification_bounds_vs_empirical(art_warning, fix_get_mnist_data):
    """
    Create adversarial examples and assert that the lower bounds given by the classifier is higher than
    the empirical attacks and vice versa for the upper bounds
    """
    from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    ptc.model.to(device)

    optimizer = torch.optim.Adam(ptc.model.parameters(), lr=0.0001)

    box_model = PyTorchIBPClassifier(
        model=ptc.model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    mnist_data = fix_get_mnist_data[0]
    mnist_labels = fix_get_mnist_data[1]

    interval_preds = box_model.predict_intervals(x=mnist_data, bounds=0.3, limits=[0.0, 1.0])
    box_model.model.set_forward_mode("attack")
    attack = ProjectedGradientDescent(
        estimator=box_model,
        eps=0.3,
        eps_step=0.001,
        max_iter=20,
        num_random_init=1,
    )
    i_batch = attack.generate(mnist_data.astype("float32"), mnist_labels.astype("float32"))
    adv_preds = box_model.predict(i_batch)

    for adv_pred, cert_prd in zip(adv_preds, interval_preds):
        assert np.all(adv_pred > cert_prd[0])
        assert np.all(adv_pred < cert_prd[1])


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_mnist_certification_training(art_warning, fix_get_mnist_data):
    """
    Assert that the training loop runs without errors
    """
    from art.defences.trainer import AdversarialTrainerCertifiedIBPPyTorch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    ptc.model.to(device)

    optimizer = torch.optim.Adam(ptc.model.parameters(), lr=0.0001)

    box_model = PyTorchIBPClassifier(
        model=ptc.model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    mnist_data = fix_get_mnist_data[0]
    mnist_labels = torch.tensor(fix_get_mnist_data[1])

    from torch.optim.lr_scheduler import MultiStepLR

    scheduler = MultiStepLR(box_model._optimizer, milestones=[2, 5], gamma=0.1)
    trainer = AdversarialTrainerCertifiedIBPPyTorch(classifier=box_model, bound=0.2)
    trainer.fit(x=mnist_data, y=mnist_labels, scheduler=scheduler, batch_size=32, nb_epochs=2, limits=[0, 1])


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_mnist_certification_training_with_pgd(art_warning, fix_get_mnist_data):
    """
    Assert that the training loop runs without errors when also doing PGD augmentation
    """
    from art.defences.trainer import AdversarialTrainerCertifiedIBPPyTorch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    ptc.model.to(device)

    optimizer = torch.optim.Adam(ptc.model.parameters(), lr=0.0001)

    box_model = PyTorchIBPClassifier(
        model=ptc.model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    mnist_data = fix_get_mnist_data[0]
    mnist_labels = torch.tensor(fix_get_mnist_data[1])

    from torch.optim.lr_scheduler import MultiStepLR

    scheduler = MultiStepLR(box_model._optimizer, milestones=[2, 5], gamma=0.1)
    trainer = AdversarialTrainerCertifiedIBPPyTorch(
        classifier=box_model,
        bound=0.2,
        augment_with_pgd=True,
        pgd_params={"eps": 0.3, "eps_step": 0.05, "max_iter": 20, "batch_size": 32, "num_random_init": 1},
    )
    trainer.fit(x=mnist_data, y=mnist_labels, scheduler=scheduler, batch_size=32, nb_epochs=2, limits=[0, 1])
