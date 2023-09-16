# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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

logger = logging.getLogger(__name__)


@pytest.fixture()
def get_pytorch_object_detector(get_default_mnist_subset):
    """
    This class tests the PyTorchObjectDetector object detector.
    """
    import torch
    import torchvision

    from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

    # Define object detector
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01)

    object_detector = PyTorchObjectDetector(
        model=model,
        input_shape=(32, 32, 3),
        optimizer=optimizer,
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


@pytest.fixture()
def get_pytorch_object_detector_mask(get_default_mnist_subset):
    """
    This class tests the PyTorchObjectDetector object detector.
    """
    import torch
    import torchvision

    from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

    # Define object detector
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01)

    object_detector = PyTorchObjectDetector(
        model=model,
        input_shape=(32, 32, 3),
        optimizer=optimizer,
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
            "masks": result[0]["masks"],
        },
        {
            "boxes": result[1]["boxes"],
            "labels": result[1]["labels"],
            "scores": np.ones_like(result[1]["labels"]),
            "masks": result[0]["masks"],
        },
    ]

    yield object_detector, x_test, y_test


@pytest.fixture()
def get_pytorch_faster_rcnn(get_default_mnist_subset):
    """
    This class tests the PyTorchFasterRCNN object detector.
    """
    import torch

    from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN

    # Define object detector
    object_detector = PyTorchFasterRCNN(
        input_shape=(32, 32, 3),
        clip_values=(0, 1),
        channels_first=False,
        attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
    )
    params = [p for p in object_detector.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01)

    object_detector.set_params(optimizer=optimizer)

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
