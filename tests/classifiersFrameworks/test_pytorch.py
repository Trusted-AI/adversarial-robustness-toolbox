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
import numpy as np
import pytest

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sklearn.datasets

from art.estimators.classification.pytorch import PyTorchClassifier
from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
from art.defences.preprocessor.spatial_smoothing_pytorch import SpatialSmoothingPyTorch
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

from tests.attacks.utils import backend_test_defended_images
from tests.utils import ARTTestException


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


# A generic test for various preprocessing_defences, forward pass.
def _test_preprocessing_defences_forward(
    get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
):
    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier_, _ = image_dl_estimator()

    clip_values = (0, 1)
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        clip_values=clip_values,
        model=classifier_.model,
        preprocessing_defences=preprocessing_defences,
        loss=criterion,
        input_shape=(1, 28, 28),
        nb_classes=10,
        device_type=device_type,
    )

    with torch.no_grad():
        predictions_classifier = classifier.predict(x_test_mnist)

    # Apply the same defences by hand
    x_test_defense = x_test_mnist
    for defence in preprocessing_defences:
        x_test_defense, _ = defence(x_test_defense, y_test_mnist)

    x_test_defense = torch.tensor(x_test_defense)
    with torch.no_grad():
        predictions_check = classifier_.model(x_test_defense)
    predictions_check = predictions_check.cpu().numpy()

    # Check that the prediction results match
    np.testing.assert_array_almost_equal(predictions_classifier, predictions_check, decimal=4)


# A generic test for various preprocessing_defences, backward pass.
def _test_preprocessing_defences_backward(
    get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
):
    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier_, _ = image_dl_estimator()

    clip_values = (0, 1)
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        clip_values=clip_values,
        model=classifier_.model,
        preprocessing_defences=preprocessing_defences,
        loss=criterion,
        input_shape=(1, 28, 28),
        nb_classes=10,
        device_type=device_type,
    )

    # The efficient defence-chaining.
    pseudo_gradients = np.random.randn(*x_test_mnist.shape)
    gradients_in_chain = classifier._apply_preprocessing_gradient(x_test_mnist, pseudo_gradients)

    # Apply the same backward pass one by one.
    x = x_test_mnist
    x_intermediates = [x]
    for preprocess in classifier.preprocessing_operations[:-1]:
        x = preprocess(x)[0]
        x_intermediates.append(x)

    gradients = pseudo_gradients
    for preprocess, x in zip(classifier.preprocessing_operations[::-1], x_intermediates[::-1]):
        gradients = preprocess.estimate_gradient(x, gradients)

    np.testing.assert_array_almost_equal(gradients_in_chain, gradients, decimal=4)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_nodefence(art_warning, get_default_mnist_subset, image_dl_estimator, device_type):
    try:
        preprocessing_defences = []
        _test_preprocessing_defences_forward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
        _test_preprocessing_defences_backward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_defence_pytorch(art_warning, get_default_mnist_subset, image_dl_estimator, device_type):
    try:
        smooth_3x3 = SpatialSmoothingPyTorch(window_size=3, channels_first=True, device_type=device_type)
        preprocessing_defences = [smooth_3x3]
        _test_preprocessing_defences_forward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
        _test_preprocessing_defences_backward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_defence_non_pytorch(art_warning, get_default_mnist_subset, image_dl_estimator, device_type):
    try:
        smooth_3x3 = SpatialSmoothing(window_size=3, channels_first=True)
        preprocessing_defences = [smooth_3x3]
        _test_preprocessing_defences_forward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
        _test_preprocessing_defences_backward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.xfail(reason="Preprocessing-defence chaining only supports defences implemented in PyTorch.")
@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_defences_pytorch_and_nonpytorch(art_warning, get_default_mnist_subset, image_dl_estimator, device_type):
    try:
        smooth_3x3_nonpth = SpatialSmoothing(window_size=3, channels_first=True)
        smooth_3x3_pth = SpatialSmoothingPyTorch(window_size=3, channels_first=True, device_type=device_type)
        preprocessing_defences = [smooth_3x3_nonpth, smooth_3x3_pth]
        _test_preprocessing_defences_forward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
        _test_preprocessing_defences_backward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_defences_chaining(art_warning, get_default_mnist_subset, image_dl_estimator, device_type):
    try:
        smooth_3x3 = SpatialSmoothingPyTorch(window_size=3, channels_first=True, device_type=device_type)
        smooth_5x5 = SpatialSmoothingPyTorch(window_size=5, channels_first=True, device_type=device_type)
        smooth_7x7 = SpatialSmoothingPyTorch(window_size=7, channels_first=True, device_type=device_type)
        preprocessing_defences = [smooth_3x3, smooth_5x5, smooth_7x7]
        _test_preprocessing_defences_forward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
        _test_preprocessing_defences_backward(
            get_default_mnist_subset, image_dl_estimator, device_type, preprocessing_defences
        )
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_fgsm_defences(art_warning, fix_get_mnist_subset, image_dl_estimator, device_type):
    try:
        clip_values = (0, 1)
        smooth_3x3 = SpatialSmoothingPyTorch(window_size=3, channels_first=True, device_type=device_type)
        smooth_5x5 = SpatialSmoothingPyTorch(window_size=5, channels_first=True, device_type=device_type)
        smooth_7x7 = SpatialSmoothingPyTorch(window_size=7, channels_first=True, device_type=device_type)
        classifier_, _ = image_dl_estimator()

        criterion = nn.CrossEntropyLoss()
        classifier = PyTorchClassifier(
            clip_values=clip_values,
            model=classifier_.model,
            preprocessing_defences=[smooth_3x3, smooth_5x5, smooth_7x7],
            loss=criterion,
            input_shape=(1, 28, 28),
            nb_classes=10,
            device_type=device_type,
        )
        assert len(classifier.preprocessing_defences) == 3

        attack = FastGradientMethod(classifier, eps=1.0, batch_size=128)
        backend_test_defended_images(attack, fix_get_mnist_subset)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_pytorch_binary_pgd(art_warning, get_mnist_dataset):
    """
    This test instantiates a binary classification PyTorch model, then attacks it using PGD

    """

    class BasicModel(nn.Module):
        def __init__(self):
            super(BasicModel, self).__init__()
            self.layer_1 = nn.Linear(20, 32)
            self.layer_2 = nn.Linear(32, 1)

        def forward(self, x):
            x = F.relu(self.layer_1(x))
            x = torch.sigmoid(self.layer_2(x))

            return x

    try:
        device = "cpu"
        x, y = sklearn.datasets.make_classification(
            n_samples=10000, n_features=20, n_informative=5, n_redundant=2, n_repeated=0, n_classes=2
        )
        train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
        train_x = test_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        test_x = test_x.astype(np.float32)
        model = BasicModel()
        loss_func = nn.BCELoss()
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        classifier = PyTorchClassifier(
            model=model,
            loss=loss_func,
            optimizer=opt,
            input_shape=(1, 28, 28),
            nb_classes=2,
        )
        classifier.fit(train_x, train_y, batch_size=64, nb_epochs=3)
        test_x_batch = test_x[0:16]
        preds = classifier.predict(test_x_batch)
        attacker = ProjectedGradientDescent(classifier, eps=0.5)
        generated = attacker.generate(test_x_batch)
        adv_predicted = classifier.predict(generated)
        assert (adv_predicted != preds).all()
    except ARTTestException as e:
        art_warning(e)
