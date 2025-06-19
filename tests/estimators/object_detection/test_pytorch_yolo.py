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


@pytest.mark.only_with_platform("pytorch")
def test_predict(art_warning, get_pytorch_yolo):
    try:
        from art.utils import non_maximum_suppression

        object_detector, x_test, _ = get_pytorch_yolo

        preds = object_detector.predict(x_test)
        result = non_maximum_suppression(preds[0], iou_threshold=0.4, confidence_threshold=0.3)
        assert list(result.keys()) == ["boxes", "labels", "scores"]

        assert result["boxes"].shape == (1, 4)
        expected_detection_boxes = np.asarray([[19.709427, 39.02864, 402.08032, 383.65576]])
        np.testing.assert_array_almost_equal(result["boxes"], expected_detection_boxes, decimal=3)

        assert result["scores"].shape == (1,)
        expected_detection_scores = np.asarray([0.40862876])
        np.testing.assert_array_almost_equal(result["scores"], expected_detection_scores, decimal=3)

        assert result["labels"].shape == (1,)
        expected_detection_classes = np.asarray([23])
        np.testing.assert_array_equal(result["labels"], expected_detection_classes)

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

        assert grads.shape == (1, 3, 416, 416)

        expected_gradients1 = np.asarray(
            [
                -7.8263599e-04,
                -3.2761338e-04,
                -1.7732104e-04,
                5.0963718e-07,
                -1.2021367e-04,
                1.5550642e-05,
                -1.2371356e-04,
                6.0041926e-05,
                7.2321229e-05,
                2.8970995e-04,
                3.2069255e-04,
                -9.7214943e-06,
                4.1050217e-04,
                3.4139317e-04,
                3.2144223e-04,
                8.0305658e-04,
                1.0029323e-03,
                5.4904580e-04,
                3.4701737e-04,
                9.2334412e-05,
                4.5694585e-05,
                -4.1882982e-04,
                -1.1162873e-03,
                -1.2383220e-03,
                -1.2119032e-03,
                -1.3792568e-03,
                -1.0219158e-03,
                -1.7796915e-04,
                1.6578102e-04,
                -4.0390861e-04,
                5.0578610e-04,
                3.2289932e-05,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 208, 192:224], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                4.02656849e-04,
                1.32368109e-03,
                1.06753211e-03,
                1.02746498e-03,
                4.34952060e-04,
                1.30278734e-03,
                1.65620341e-03,
                8.48031021e-04,
                2.80185544e-04,
                2.04326061e-04,
                -9.31014947e-05,
                -4.90375911e-04,
                -3.42604442e-04,
                1.36689676e-04,
                3.08552640e-04,
                3.88148270e-04,
                1.00293232e-03,
                -1.08163455e-04,
                -1.41605944e-03,
                -1.96112506e-03,
                -6.27453031e-04,
                -9.53144976e-04,
                -6.66696171e-04,
                -5.78872336e-04,
                -1.52492896e-04,
                -1.06580940e-03,
                1.04899483e-03,
                5.83183893e-04,
                8.98627564e-04,
                3.37607635e-04,
                8.34865321e-04,
                5.12865488e-04,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 192:224, 208], expected_gradients2, decimal=2)

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

        assert grads.shape == (1, 3, 416, 416)

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

        assert pytest.approx(0.0920641, abs=0.05) == float(loss)

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


@pytest.mark.only_with_platform("pytorch")
def test_patch(art_warning, get_pytorch_yolo):
    try:

        from art.attacks.evasion import AdversarialPatchPyTorch

        rotation_max = 0.0
        scale_min = 0.1
        scale_max = 0.3
        distortion_scale_max = 0.0
        learning_rate = 1.99
        max_iter = 2
        batch_size = 16
        patch_shape = (3, 5, 5)
        patch_type = "circle"
        optimizer = "pgd"

        object_detector, x_test, y_test = get_pytorch_yolo

        ap = AdversarialPatchPyTorch(
            estimator=object_detector,
            rotation_max=rotation_max,
            scale_min=scale_min,
            scale_max=scale_max,
            optimizer=optimizer,
            distortion_scale_max=distortion_scale_max,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            patch_shape=patch_shape,
            patch_type=patch_type,
            verbose=True,
            targeted=False,
        )

        _, _ = ap.generate(x=x_test, y=y_test)

        patched_images = ap.apply_patch(x_test, scale=0.4)
        result = object_detector.predict(patched_images)

        assert result[0]["scores"].shape == (10647,)
        expected_detection_scores = np.asarray(
            [
                2.0058684e-08,
                8.2879878e-06,
                1.5323505e-05,
                8.5337388e-06,
                1.5668766e-05,
                3.7196922e-05,
                4.5348370e-05,
                6.9575308e-06,
                4.2298670e-06,
                1.0316832e-06,
            ]
        )
        np.testing.assert_allclose(result[0]["scores"][:10], expected_detection_scores, rtol=1e-5, atol=1e-8)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_translate_predictions_yolov8_format():
    import torch
    import numpy as np
    from art.estimators.object_detection.pytorch_yolo import PyTorchYolo

    # Create a dummy PyTorchYolo instance (model is not used for this test)
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x
    dummy_model = DummyModel()
    yolo = PyTorchYolo(
        model=dummy_model,
        input_shape=(3, 416, 416),
        optimizer=None,
        clip_values=(0, 1),
        channels_first=True,
        attack_losses=("loss_total",),
    )

    # Mock YOLO v8+ style predictions: list of dicts with torch tensors
    pred_boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)
    pred_labels = torch.tensor([5], dtype=torch.int64)
    pred_scores = torch.tensor([0.9], dtype=torch.float32)
    predictions = [{
        "boxes": pred_boxes,
        "labels": pred_labels,
        "scores": pred_scores,
    }]

    # Call the translation method
    translated = yolo._translate_predictions(predictions)

    # Check output type and values
    assert isinstance(translated, list)
    assert isinstance(translated[0], dict)
    assert isinstance(translated[0]["boxes"], np.ndarray)
    assert isinstance(translated[0]["labels"], np.ndarray)
    assert isinstance(translated[0]["scores"], np.ndarray)
    np.testing.assert_array_equal(translated[0]["boxes"], pred_boxes.numpy())
    np.testing.assert_array_equal(translated[0]["labels"], pred_labels.numpy())
    np.testing.assert_array_equal(translated[0]["scores"], pred_scores.numpy())


@pytest.mark.only_with_platform("pytorch")
def test_pytorch_yolo_loss_wrapper_additional_losses():
    import torch
    from art.estimators.object_detection.pytorch_yolo import PyTorchYoloLossWrapper

    # Dummy model with a .loss() method
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def loss(self, items):
            # Return (loss, [loss_box, loss_cls, loss_dfl])
            return (
                torch.tensor([1.0, 2.0, 3.0]),
                [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
            )

    dummy_model = DummyModel()
    # Patch ultralytics import in the wrapper
    import sys
    import types
    ultralytics_mock = types.SimpleNamespace(
        models=types.SimpleNamespace(yolo=types.SimpleNamespace(detect=types.SimpleNamespace(DetectionPredictor=lambda: types.SimpleNamespace(args=None)))),
        utils=types.SimpleNamespace(loss=types.SimpleNamespace(v8DetectionLoss=lambda m: None, E2EDetectLoss=lambda m: None))
    )
    sys.modules['ultralytics'] = ultralytics_mock
    sys.modules['ultralytics.models'] = ultralytics_mock.models
    sys.modules['ultralytics.models.yolo'] = ultralytics_mock.models.yolo
    sys.modules['ultralytics.models.yolo.detect'] = ultralytics_mock.models.yolo.detect
    sys.modules['ultralytics.utils'] = ultralytics_mock.utils
    sys.modules['ultralytics.utils.loss'] = ultralytics_mock.utils.loss

    wrapper = PyTorchYoloLossWrapper(dummy_model, name="yolov8n")
    wrapper.train()
    # Dummy input and targets
    x = torch.zeros((1, 3, 416, 416))
    targets = [{"boxes": torch.zeros((1, 4)), "labels": torch.zeros((1,))}]
    losses = wrapper(x, targets)
    assert set(losses.keys()) == {"loss_total", "loss_box", "loss_cls", "loss_dfl"}
    assert losses["loss_total"].item() == 6.0  # sum([1.0, 2.0, 3.0])
    assert losses["loss_box"].item() == 1.0
    assert losses["loss_cls"].item() == 2.0
    assert losses["loss_dfl"].item() == 3.0
