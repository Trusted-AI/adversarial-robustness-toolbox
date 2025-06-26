# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
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
import os
from art.estimators.object_detection.pytorch_yolo_loss_wrapper import PyTorchYoloLossWrapper
from ultralytics import YOLO

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


@pytest.mark.only_with_platform("pytorch")
def test_yolov8_loss_wrapper():
    """Test the loss wrapper with YOLOv8 model."""
    # Load YOLOv8 model
    model_path = "/tmp/yolo_v8.3.0/yolov8n.pt"
    model = YOLO(model_path).model

    # Create wrapper
    wrapper = PyTorchYoloLossWrapper(model, name="yolov8n")
    wrapper.train()

    # Create sample input
    batch_size = 2
    x = torch.randn((batch_size, 3, 640, 640))  # YOLOv8 expects (B, 3, 640, 640)

    # Create targets
    targets = []
    for _ in range(batch_size):
        boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]])  # [x1, y1, x2, y2]
        labels = torch.zeros(2, dtype=torch.long)  # Use class 0 for testing
        targets.append({"boxes": boxes, "labels": labels})

    # Test training mode
    losses = wrapper(x, targets)

    # Validate loss structure
    expected_loss_keys = {"loss_total", "loss_box", "loss_cls", "loss_dfl"}
    assert set(losses.keys()) == expected_loss_keys
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert all(not torch.isnan(v).any() for v in losses.values()), "Loss values contain NaN"
    assert all(not torch.isinf(v).any() for v in losses.values()), "Loss values contain Inf"

    # Test inference mode
    wrapper.eval()
    with torch.no_grad():
        predictions = wrapper(x)

    # Validate predictions
    assert isinstance(predictions, list)
    assert len(predictions) == batch_size
    for pred in predictions:
        assert set(pred.keys()) == {"boxes", "scores", "labels"}
        assert isinstance(pred["boxes"], torch.Tensor)
        assert isinstance(pred["scores"], torch.Tensor)
        assert isinstance(pred["labels"], torch.Tensor)
        assert pred["boxes"].ndim == 2 and pred["boxes"].shape[1] == 4
        assert pred["scores"].ndim == 1
        assert pred["labels"].ndim == 1
        assert pred["scores"].shape[0] == pred["labels"].shape[0] == pred["boxes"].shape[0]
        assert pred["boxes"].dtype == torch.float32
        assert pred["labels"].dtype in (torch.int32, torch.int64)


@pytest.mark.only_with_platform("pytorch")
def test_yolov10_loss_wrapper():
    """Test the loss wrapper with YOLOv10 model."""
    # Load YOLOv10 model
    model_path = "/tmp/yolo_v8.3.0/yolov10n.pt"
    model = YOLO(model_path).model

    # Create wrapper
    wrapper = PyTorchYoloLossWrapper(model, name="yolov10n")
    wrapper.train()

    # Create sample input
    batch_size = 2
    x = torch.randn((batch_size, 3, 640, 640))  # Standard YOLO input size

    # Create targets
    targets = []
    for _ in range(batch_size):
        boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]])  # [x1, y1, x2, y2]
        labels = torch.zeros(2, dtype=torch.long)  # Use class 0 for testing
        targets.append({"boxes": boxes, "labels": labels})

    # Test training mode
    losses = wrapper(x, targets)

    # Validate loss structure
    expected_loss_keys = {"loss_total", "loss_box", "loss_cls", "loss_dfl"}
    assert set(losses.keys()) == expected_loss_keys
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert all(not torch.isnan(v).any() for v in losses.values()), "Loss values contain NaN"
    assert all(not torch.isinf(v).any() for v in losses.values()), "Loss values contain Inf"
    assert all(v.item() >= 0 for v in losses.values()), "Loss values should be non-negative"
    assert losses["loss_total"].item() > 0, "Total loss should be positive"

    # Test inference mode
    wrapper.eval()
    with torch.no_grad():
        predictions = wrapper(x)

    # Validate predictions
    assert isinstance(predictions, list)
    assert len(predictions) == batch_size
    for pred in predictions:
        assert set(pred.keys()) == {"boxes", "scores", "labels"}
        assert isinstance(pred["boxes"], torch.Tensor)
        assert isinstance(pred["scores"], torch.Tensor)
        assert isinstance(pred["labels"], torch.Tensor)
        assert pred["boxes"].ndim == 2 and pred["boxes"].shape[1] == 4
        assert pred["scores"].ndim == 1
        assert pred["labels"].ndim == 1
        assert pred["scores"].shape[0] == pred["labels"].shape[0] == pred["boxes"].shape[0]
        assert pred["boxes"].dtype == torch.float32
        assert pred["labels"].dtype in (torch.int32, torch.int64)


@pytest.mark.only_with_platform("pytorch")
def test_translate_predictions_yolov8_format():
    import torch
    import numpy as np
    from art.estimators.object_detection.pytorch_yolo import PyTorchYolo

    # Create a dummy PyTorchYolo instance (model is not used for this test)
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    test_model = DummyModel()
    yolo = PyTorchYolo(
        model=test_model,
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
    predictions = [
        {
            "boxes": pred_boxes,
            "labels": pred_labels,
            "scores": pred_scores,
        }
    ]

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

    # Dummy model with a .loss() method
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def loss(self, items):
            # Return (loss, [loss_box, loss_cls, loss_dfl])
            return (torch.tensor([1.0, 2.0, 3.0]), [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])

    test_model = DummyModel()
    # Patch ultralytics import in the wrapper
    import sys
    import types

    ultralytics_mock = types.SimpleNamespace(
        models=types.SimpleNamespace(
            yolo=types.SimpleNamespace(
                detect=types.SimpleNamespace(DetectionPredictor=lambda: types.SimpleNamespace(args=None))
            )
        ),
        utils=types.SimpleNamespace(
            loss=types.SimpleNamespace(v8DetectionLoss=lambda m: None, E2EDetectLoss=lambda m: None)
        ),
    )
    sys.modules["ultralytics"] = ultralytics_mock
    sys.modules["ultralytics.models"] = ultralytics_mock.models
    sys.modules["ultralytics.models.yolo"] = ultralytics_mock.models.yolo
    sys.modules["ultralytics.models.yolo.detect"] = ultralytics_mock.models.yolo.detect
    sys.modules["ultralytics.utils"] = ultralytics_mock.utils
    sys.modules["ultralytics.utils.loss"] = ultralytics_mock.utils.loss

    wrapper = PyTorchYoloLossWrapper(test_model, name="yolov8n")
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


@pytest.mark.only_with_platform("pytorch")
def test_loss_wrapper_outputs_all_losses():
    # Dummy model with a .loss() method
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def loss(self, items):
            # Return (loss, [loss_box, loss_cls, loss_dfl])
            return (torch.tensor([1.0, 2.0, 3.0]), [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])

    test_model = DummyModel()
    # Patch ultralytics import in the wrapper
    import sys
    import types

    ultralytics_mock = types.SimpleNamespace(
        models=types.SimpleNamespace(
            yolo=types.SimpleNamespace(
                detect=types.SimpleNamespace(DetectionPredictor=lambda: types.SimpleNamespace(args=None))
            )
        ),
        utils=types.SimpleNamespace(
            loss=types.SimpleNamespace(v8DetectionLoss=lambda m: None, E2EDetectLoss=lambda m: None)
        ),
    )
    sys.modules["ultralytics"] = ultralytics_mock
    sys.modules["ultralytics.models"] = ultralytics_mock.models
    sys.modules["ultralytics.models.yolo"] = ultralytics_mock.models.yolo
    sys.modules["ultralytics.models.yolo.detect"] = ultralytics_mock.models.yolo.detect
    sys.modules["ultralytics.utils"] = ultralytics_mock.utils
    sys.modules["ultralytics.utils.loss"] = ultralytics_mock.utils.loss

    wrapper = PyTorchYoloLossWrapper(test_model, name="yolov8n")
    wrapper.train()
    x = torch.zeros((1, 3, 416, 416))
    targets = [{"boxes": torch.zeros((1, 4)), "labels": torch.zeros((1,))}]
    losses = wrapper(x, targets)
    assert set(losses.keys()) == {"loss_total", "loss_box", "loss_cls", "loss_dfl"}
    assert losses["loss_total"].item() == 6.0
    assert losses["loss_box"].item() == 1.0
    assert losses["loss_cls"].item() == 2.0
    assert losses["loss_dfl"].item() == 3.0


@pytest.mark.only_with_platform("pytorch")
def test_yolov8_vs_yolov10_loss_functions():
    import torch

    # Dummy model that will be wrapped
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = None  # Will be set by wrapper

        def loss(self, items):
            # Return different loss components based on model version
            return (torch.tensor([1.0, 2.0, 3.0]), [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])

    # Mock ultralytics imports
    import sys
    import types

    def create_mock_imports():
        return types.SimpleNamespace(
            models=types.SimpleNamespace(
                yolo=types.SimpleNamespace(
                    detect=types.SimpleNamespace(DetectionPredictor=lambda: types.SimpleNamespace(args=None))
                )
            ),
            utils=types.SimpleNamespace(
                loss=types.SimpleNamespace(v8DetectionLoss=lambda m: "v8_loss", E2EDetectLoss=lambda m: "v10_loss")
            ),
        )

    # Test YOLOv8
    ultralytics_mock = create_mock_imports()
    sys.modules["ultralytics"] = ultralytics_mock
    sys.modules["ultralytics.models"] = ultralytics_mock.models
    sys.modules["ultralytics.models.yolo"] = ultralytics_mock.models.yolo
    sys.modules["ultralytics.models.yolo.detect"] = ultralytics_mock.models.yolo.detect
    sys.modules["ultralytics.utils"] = ultralytics_mock.utils
    sys.modules["ultralytics.utils.loss"] = ultralytics_mock.utils.loss

    model_v8 = DummyModel()
    wrapper_v8 = PyTorchYoloLossWrapper(model_v8, name="yolov8n")
    assert wrapper_v8.model.criterion == "v8_loss"

    # Test YOLOv10
    model_v10 = DummyModel()
    wrapper_v10 = PyTorchYoloLossWrapper(model_v10, name="yolov10n")
    assert wrapper_v10.model.criterion == "v10_loss"


@pytest.mark.only_with_platform("pytorch")
def test_yolov8_inference_mode():
    import torch
    import numpy as np

    # Dummy model that mimics YOLO v8+ inference behavior
    class DummyYoloV8Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Return format matching YOLO v8+ output structure
            return [{"boxes": torch.ones(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1)}]

    # Mock ultralytics imports
    import sys
    import types

    ultralytics_mock = types.SimpleNamespace(
        models=types.SimpleNamespace(
            yolo=types.SimpleNamespace(
                detect=types.SimpleNamespace(
                    DetectionPredictor=lambda: types.SimpleNamespace(
                        args=None,
                        model=None,
                        batch=None,
                        postprocess=lambda preds, *args: [
                            types.SimpleNamespace(
                                boxes=types.SimpleNamespace(
                                    xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                                    conf=torch.tensor([0.95]),
                                    cls=torch.tensor([1]),
                                )
                            )
                        ],
                    )
                )
            )
        ),
        utils=types.SimpleNamespace(
            loss=types.SimpleNamespace(v8DetectionLoss=lambda m: None, E2EDetectLoss=lambda m: None)
        ),
    )
    sys.modules["ultralytics"] = ultralytics_mock
    sys.modules["ultralytics.models"] = ultralytics_mock.models
    sys.modules["ultralytics.models.yolo"] = ultralytics_mock.models.yolo
    sys.modules["ultralytics.models.yolo.detect"] = ultralytics_mock.models.yolo.detect
    sys.modules["ultralytics.utils"] = ultralytics_mock.utils
    sys.modules["ultralytics.utils.loss"] = ultralytics_mock.utils.loss

    model = DummyYoloV8Model()
    wrapper = PyTorchYoloLossWrapper(model, name="yolov8n")
    wrapper.eval()  # Set to inference mode

    # Test inference
    x = torch.zeros((1, 3, 416, 416))
    predictions = wrapper(x)

    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert set(predictions[0].keys()) == {"boxes", "scores", "labels"}
    assert isinstance(predictions[0]["boxes"], torch.Tensor)
    assert isinstance(predictions[0]["scores"], torch.Tensor)
    assert isinstance(predictions[0]["labels"], torch.Tensor)
    np.testing.assert_array_equal(predictions[0]["boxes"].numpy(), np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    np.testing.assert_array_equal(predictions[0]["scores"].numpy(), np.array([0.95], dtype=np.float32))
    np.testing.assert_array_equal(predictions[0]["labels"].numpy(), np.array([1], dtype=np.int64))


@pytest.mark.only_with_platform("pytorch")
def test_yolov8_training_data_format():
    import torch

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def loss(self, items):
            # Validate input format matches expected YOLO v8+ training format
            assert "bboxes" in items
            assert "cls" in items
            assert "batch_idx" in items
            assert "img" in items
            return torch.tensor([1.0]), [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)]

    # Setup mock imports
    import sys
    import types

    ultralytics_mock = types.SimpleNamespace(
        models=types.SimpleNamespace(
            yolo=types.SimpleNamespace(
                detect=types.SimpleNamespace(DetectionPredictor=lambda: types.SimpleNamespace(args=None))
            )
        ),
        utils=types.SimpleNamespace(
            loss=types.SimpleNamespace(v8DetectionLoss=lambda m: None, E2EDetectLoss=lambda m: None)
        ),
    )
    sys.modules["ultralytics"] = ultralytics_mock
    sys.modules["ultralytics.models"] = ultralytics_mock.models
    sys.modules["ultralytics.models.yolo"] = ultralytics_mock.models.yolo
    sys.modules["ultralytics.models.yolo.detect"] = ultralytics_mock.models.yolo.detect
    sys.modules["ultralytics.utils"] = ultralytics_mock.utils
    sys.modules["ultralytics.utils.loss"] = ultralytics_mock.utils.loss

    model = DummyModel()
    wrapper = PyTorchYoloLossWrapper(model, name="yolov8n")
    wrapper.train()

    # Test with different batch sizes and box counts
    batch_sizes = [1, 2]
    box_counts = [1, 3]

    for batch_size in batch_sizes:
        for box_count in box_counts:
            x = torch.zeros((batch_size, 3, 416, 416))
            targets = [
                {"boxes": torch.zeros((box_count, 4)), "labels": torch.zeros(box_count)} for _ in range(batch_size)
            ]
            losses = wrapper(x, targets)

            # Verify loss structure
            assert set(losses.keys()) == {"loss_total", "loss_box", "loss_cls", "loss_dfl"}
            assert all(isinstance(v, torch.Tensor) for v in losses.values())
