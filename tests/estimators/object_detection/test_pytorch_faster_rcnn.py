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

import logging

import numpy as np
import pytest

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def get_pytorch_faster_rcnn(get_default_mnist_subset):
    """
    This class tests the PyTorchFasterRCNN object detector.
    """
    from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN

    # Define object detector
    object_detector = PyTorchFasterRCNN(
        clip_values=(0, 1),
        channels_first=False,
        attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
    )

    (_, _), (x_test_mnist, _) = get_default_mnist_subset

    x_test = np.transpose(x_test_mnist[:2], (0, 2, 3, 1))
    x_test = np.repeat(x_test.astype(np.float32), repeats=3, axis=3)

    # Create labels
    result = object_detector.predict(x=x_test)

    y_test = [
        {
            "boxes": result[0]["boxes"],
            "labels": result[0]["labels"],
            "scores": np.ones_like(result[0]["labels"]),
        },
        {
            "boxes": result[1]["boxes"],
            "labels": result[1]["labels"],
            "scores": np.ones_like(result[1]["labels"]),
        },
    ]

    yield object_detector, x_test, y_test


@pytest.mark.only_with_platform("pytorch")
def test_predict(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, _ = get_pytorch_faster_rcnn

        result = object_detector.predict(x_test)
        assert list(result[0].keys()) == ["boxes", "labels", "scores"]

        assert result[0]["boxes"].shape == (7, 4)
        expected_detection_boxes = np.asarray([4.4017954, 6.3090835, 22.128296, 27.570665])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        assert result[0]["scores"].shape == (7,)
        expected_detection_scores = np.asarray(
            [0.3314798, 0.14125851, 0.13928168, 0.0996184, 0.08550017, 0.06690315, 0.05359321]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

        assert result[0]["labels"].shape == (7,)
        expected_detection_classes = np.asarray([72, 79, 1, 72, 78, 72, 82])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_fit(art_warning, get_pytorch_faster_rcnn):
    try:
        import torch

        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        params = [p for p in object_detector.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.01)

        object_detector.set_params(optimizer=optimizer)

        # Compute loss before training
        loss1 = object_detector.compute_loss(x=x_test, y=y_test)

        # Train for one epoch
        object_detector.fit(x_test, y_test, nb_epochs=1)

        # Compute loss after training
        loss2 = object_detector.compute_loss(x=x_test, y=y_test)

        assert loss1 != loss2

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_loss_gradient(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        # Compute gradients
        grads = object_detector.loss_gradient(x_test, y_test)
        assert grads.shape == (2, 28, 28, 3)

        expected_gradients1 = np.asarray(
            [
                [5.7717459e-04, 2.2427551e-03, 2.7338031e-03],
                [-5.4135895e-04, -6.8901619e-03, -5.3023611e-04],
                [1.7901474e-03, -6.0165934e-03, 1.2608932e-03],
                [2.2302025e-03, -4.1366839e-03, 8.1665488e-04],
                [5.0025941e-03, -2.0607577e-03, 1.3738470e-03],
                [6.7711552e-03, 2.4779334e-03, 3.2517519e-03],
                [7.7946498e-03, 3.8083603e-03, 3.9150072e-03],
                [6.2914360e-03, 3.2317259e-03, 2.4392023e-03],
                [6.8533504e-03, 4.6805567e-03, 2.1657508e-03],
                [6.4596147e-03, 1.6440222e-03, 2.1018654e-03],
                [7.3140049e-03, 4.9051084e-03, 2.1954530e-03],
                [7.3917350e-03, 5.3877393e-03, 2.5017208e-03],
                [7.1420427e-03, 4.5424267e-03, 1.7418499e-03],
                [7.6933270e-03, 7.0741987e-03, 1.3693030e-03],
                [7.9037091e-03, 8.1887292e-03, 1.0207348e-03],
                [4.7930530e-03, 1.2661386e-04, -2.0549579e-03],
                [4.7417181e-03, 1.1090005e-03, -2.1967045e-03],
                [4.0628687e-03, -1.0743369e-03, -2.7016401e-03],
                [4.1211918e-03, -9.3981961e-04, -3.3123612e-03],
                [2.7677750e-03, -2.0360684e-03, -2.4159362e-03],
                [1.5355040e-03, -2.3622375e-03, -2.2277990e-03],
                [-8.2429928e-05, -2.7951330e-03, -2.4791150e-03],
                [8.6106811e-05, -1.1048347e-03, -1.8214922e-03],
                [1.3870616e-03, 1.4906849e-03, -3.1876419e-04],
                [1.1308161e-03, 6.2550785e-04, 7.9436734e-04],
                [-1.0549244e-03, -2.1480548e-03, -8.4300683e-04],
                [7.4692059e-04, 6.3713623e-04, -2.2322751e-04],
                [1.6337358e-04, -1.2138729e-03, -8.6526090e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [8.09008547e-04, 1.46970048e-03, 2.30784086e-03],
                [1.57560175e-03, -3.95192811e-03, -3.42682266e-04],
                [1.17776252e-03, -4.75858618e-03, -1.83509255e-03],
                [-3.62795522e-03, -7.03671249e-03, -2.61869049e-03],
                [-5.65498043e-03, -9.36302636e-03, -2.72479979e-03],
                [-6.13390049e-03, -1.91371012e-02, -8.64498038e-03],
                [4.13261494e-03, -5.83548984e-03, -5.41773997e-03],
                [2.10555550e-02, 1.75252277e-02, 1.19110784e-02],
                [2.86780880e-03, -2.02223212e-02, 4.42323042e-03],
                [1.66129377e-02, 4.57757805e-03, 3.99308838e-03],
                [-5.31449541e-03, -2.39533130e-02, -1.50507865e-02],
                [-1.55420639e-02, -6.57757046e-03, -1.95033997e-02],
                [-1.71425883e-02, -8.82681739e-03, -1.03681823e-02],
                [-1.52608315e-02, -2.59394385e-02, -8.74401908e-03],
                [-1.98556799e-02, -4.51070368e-02, -2.01500412e-02],
                [-1.76412370e-02, -4.00045775e-02, -2.76774243e-02],
                [-3.39970365e-02, -5.27175590e-02, -2.48762686e-02],
                [-1.01934038e-02, -1.34583283e-02, 2.92114611e-03],
                [9.27460939e-03, -1.07238982e-02, 1.69319492e-02],
                [1.32648731e-02, 7.15299882e-03, 1.81243364e-02],
                [1.04831355e-02, 3.29193124e-03, 1.09448479e-02],
                [5.21936268e-03, -1.08520268e-03, 4.44627739e-03],
                [4.43769246e-03, 1.22211361e-03, 1.76453649e-03],
                [2.82945228e-03, 1.39565568e-03, 5.05451404e-04],
                [6.36306650e-04, -7.02011574e-04, 8.36413165e-05],
                [2.80080014e-04, -9.24700813e-04, -6.42473227e-04],
                [1.44194404e-03, 9.39335907e-04, -1.95080182e-04],
                [1.05228636e-03, -4.52511711e-03, -5.74906298e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, :, :], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_errors(art_warning):
    try:
        from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN

        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(1, 2),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(-1, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                preprocessing=(0, 1),
            )

        from art.defences.postprocessor.rounded import Rounded

        post_def = Rounded()
        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                postprocessing_defences=post_def,
            )

    except ARTTestException as e:
        art_warning(e)


def test_preprocessing_defences(art_warning, get_pytorch_faster_rcnn):
    try:
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

        pre_def = SpatialSmoothing()

        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        object_detector.set_params(preprocessing_defences=pre_def)

        # Compute gradients
        grads = object_detector.loss_gradient(x=x_test, y=y_test)

        assert grads.shape == (2, 28, 28, 3)

    except ARTTestException as e:
        art_warning(e)


def test_compute_losses(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, y_test = get_pytorch_faster_rcnn
        losses = object_detector.compute_losses(x=x_test, y=y_test)
        assert len(losses) == 4

    except ARTTestException as e:
        art_warning(e)


def test_compute_loss(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        # Compute loss
        loss = object_detector.compute_loss(x=x_test, y=y_test)

        assert pytest.approx(0.84883332, abs=0.01) == float(loss)

    except ARTTestException as e:
        art_warning(e)


def test_pgd(art_warning, get_pytorch_faster_rcnn):
    try:
        from art.attacks.evasion import ProjectedGradientDescent

        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
        x_test_adv = attack.generate(x=x_test, y=y_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)

    except ARTTestException as e:
        art_warning(e)
