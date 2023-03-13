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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import pytest

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def get_pytorch_yolo(get_default_cifar10_subset):
    """
    This class tests the PyTorchYolo object detector.
    """
    import cv2
    import torch

    from pytorchyolo import models
    from pytorchyolo.utils.loss import compute_loss

    from art.estimators.object_detection.pytorch_yolo import PyTorchYolo

    model_path = "/tmp/PyTorch-YOLOv3/config/yolov3.cfg"
    weights_path = "/tmp/PyTorch-YOLOv3/weights/yolov3.weights"
    model = models.load_model(model_path=model_path, weights_path=weights_path)

    class YoloV3(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, targets=None):
            if self.training:
                outputs = self.model(x)
                # loss is averaged over a batch. Thus, for patch generation use batch_size = 1
                loss, _ = compute_loss(outputs, targets, self.model)

                loss_components = {"loss_total": loss}

                return loss_components
            else:
                return self.model(x)

    model = YoloV3(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01)

    object_detector = PyTorchYolo(
        model=model,
        input_shape=(3, 416, 416),
        optimizer=optimizer,
        clip_values=(0, 1),
        channels_first=True,
        attack_losses=("loss_total",),
    )

    (_, _), (x_test_cifar10, _) = get_default_cifar10_subset

    x_test = cv2.resize(
        x_test_cifar10[0].transpose((1, 2, 0)), dsize=(416, 416), interpolation=cv2.INTER_CUBIC
    ).transpose((2, 0, 1))
    x_test = np.expand_dims(x_test, axis=0)
    x_test = np.repeat(x_test, repeats=2, axis=0)

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
def test_predict(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, _ = get_pytorch_yolo

        result = object_detector.predict(x=x_test)

        assert list(result[0].keys()) == ["boxes", "labels", "scores"]

        assert result[0]["boxes"].shape == (10647, 4)
        expected_detection_boxes = np.asarray([0.0000000e00, 0.0000000e00, 1.6367816e02, 4.4342079e01])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        assert result[0]["scores"].shape == (10647,)
        expected_detection_scores = np.asarray(
            [
                4.3653536e-08,
                3.3987994e-06,
                2.5681820e-06,
                3.9782722e-06,
                2.1766680e-05,
                2.6138965e-05,
                6.3377396e-05,
                7.6248516e-06,
                4.3447722e-06,
                3.6515078e-06,
            ]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

        assert result[0]["labels"].shape == (10647,)
        expected_detection_classes = np.asarray([0, 0, 14, 14, 14, 14, 14, 14, 14, 0])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_fit(art_warning, get_pytorch_yolo):

    try:
        object_detector, x_test, y_test = get_pytorch_yolo

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
def test_loss_gradient(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, y_test = get_pytorch_yolo

        grads = object_detector.loss_gradient(x=x_test, y=y_test)

        assert grads.shape == (2, 3, 416, 416)

        expected_gradients1 = np.asarray(
            [
                1.28398424e-05,
                -2.95201448e-06,
                1.99362366e-05,
                -1.69057967e-05,
                2.65632730e-06,
                -9.32226249e-06,
                -9.81486028e-06,
                1.72775799e-05,
                9.19622835e-07,
                1.65521160e-05,
                1.52651919e-05,
                -4.03170270e-06,
                8.10160145e-06,
                1.99810020e-05,
                1.70234252e-05,
                -1.04990377e-05,
                -2.20760157e-05,
                2.89993841e-05,
                2.43352024e-05,
                5.40295805e-05,
                -3.54345357e-05,
                1.49476218e-05,
                -1.83201955e-05,
                -4.52892618e-06,
                -2.10271646e-05,
                -1.38741379e-05,
                1.19676406e-06,
                1.44154410e-05,
                -1.16514025e-06,
                -2.48137949e-05,
                -4.98828103e-06,
                1.53667770e-05,
                3.29377417e-06,
                2.14245338e-05,
                1.82093459e-06,
                2.11810093e-05,
                1.19740625e-05,
                1.71546981e-06,
                -1.24681810e-05,
                -7.98507535e-05,
                -3.12026459e-05,
                1.01383330e-05,
                -3.08082854e-05,
                -2.56484091e-05,
                -6.88045111e-05,
                1.62506112e-05,
                -1.15138228e-05,
                -9.07951107e-06,
                1.25368360e-05,
                -1.09746734e-05,
            ]
        )

        np.testing.assert_array_almost_equal(grads[0, 0, 208, 175:225], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                1.28398424e-05,
                -2.95201448e-06,
                1.99362366e-05,
                -1.69057967e-05,
                2.65632730e-06,
                -9.32226249e-06,
                -9.81486028e-06,
                1.72775799e-05,
                9.19622835e-07,
                1.65521160e-05,
                1.52651919e-05,
                -4.03170270e-06,
                8.10160145e-06,
                1.99810020e-05,
                1.70234252e-05,
                -1.04990377e-05,
                -2.20760157e-05,
                2.89993841e-05,
                2.43352024e-05,
                5.40295805e-05,
                -3.54345357e-05,
                1.49476218e-05,
                -1.83201955e-05,
                -4.52892618e-06,
                -2.10271646e-05,
                -1.38741379e-05,
                1.19676406e-06,
                1.44154410e-05,
                -1.16514025e-06,
                -2.48137949e-05,
                -4.98828103e-06,
                1.53667770e-05,
                3.29377417e-06,
                2.14245338e-05,
                1.82093459e-06,
                2.11810093e-05,
                1.19740625e-05,
                1.71546981e-06,
                -1.24681810e-05,
                -7.98507535e-05,
                -3.12026459e-05,
                1.01383330e-05,
                -3.08082854e-05,
                -2.56484091e-05,
                -6.88045111e-05,
                1.62506112e-05,
                -1.15138228e-05,
                -9.07951107e-06,
                1.25368360e-05,
                -1.09746734e-05,
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, 208, 175:225], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_errors(art_warning):
    try:
        from pytorchyolo import models

        from art.estimators.object_detection.pytorch_yolo import PyTorchYolo

        model_path = "/tmp/PyTorch-YOLOv3/config/yolov3.cfg"
        weights_path = "/tmp/PyTorch-YOLOv3/weights/yolov3.weights"
        model = models.load_model(model_path=model_path, weights_path=weights_path)

        with pytest.raises(ValueError):
            PyTorchYolo(
                model=model,
                clip_values=(1, 2),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        with pytest.raises(ValueError):
            PyTorchYolo(
                model=model,
                clip_values=(-1, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        from art.defences.postprocessor.rounded import Rounded

        post_def = Rounded()
        with pytest.raises(ValueError):
            PyTorchYolo(
                model=model,
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                postprocessing_defences=post_def,
            )

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_preprocessing_defences(art_warning, get_pytorch_yolo):
    try:
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

        pre_def = SpatialSmoothing()

        object_detector, x_test, y_test = get_pytorch_yolo

        object_detector.set_params(preprocessing_defences=pre_def)

        # Compute gradients
        grads = object_detector.loss_gradient(x=x_test, y=y_test)

        assert grads.shape == (2, 3, 416, 416)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_compute_losses(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, y_test = get_pytorch_yolo
        losses = object_detector.compute_losses(x=x_test, y=y_test)
        assert len(losses) == 1

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_compute_loss(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, y_test = get_pytorch_yolo

        # Compute loss
        loss = object_detector.compute_loss(x=x_test, y=y_test)

        assert pytest.approx(0.0078019718, abs=0.01) == float(loss)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_pgd(art_warning, get_pytorch_yolo):
    try:
        from art.attacks.evasion import ProjectedGradientDescent

        object_detector, x_test, y_test = get_pytorch_yolo

        attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
        x_test_adv = attack.generate(x=x_test, y=y_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)

    except ARTTestException as e:
        art_warning(e)
