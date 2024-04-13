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
def get_pytorch_object_detector_data(get_default_cifar10_subset):
    import cv2

    (_, _), (x_test_cifar10, _) = get_default_cifar10_subset
    x_test = cv2.resize(np.reshape(x_test_cifar10[0], (32, 32, 3)), dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    x_test = np.transpose(x_test, (2, 0, 1))
    x_test = np.expand_dims(x_test, axis=0)
    y_test = [
        {
            "boxes": np.asarray([[6, 22, 413, 346], [0, 24, 406, 342]], dtype=np.float32),
            "labels": np.asarray([21, 18]),
            "scores": np.asarray([1, 1], dtype=np.float32),
            "masks": np.ones((1, 416, 416)) / 2,
        }
    ]

    yield x_test, y_test


@pytest.fixture()
def get_pytorch_object_detector(get_pytorch_object_detector_data):
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
        input_shape=(3, 416, 416),
        optimizer=optimizer,
        clip_values=(0, 1),
        channels_first=True,
        attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
    )

    x_test, y_test = get_pytorch_object_detector_data

    yield object_detector, x_test, y_test


@pytest.fixture()
def get_pytorch_object_detector_mask(get_pytorch_object_detector_data):
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
        input_shape=(3, 416, 416),
        optimizer=optimizer,
        clip_values=(0, 1),
        channels_first=True,
        attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
    )

    x_test, y_test = get_pytorch_object_detector_data

    yield object_detector, x_test, y_test


@pytest.fixture()
def get_pytorch_faster_rcnn(get_pytorch_object_detector_data):
    """
    This class tests the PyTorchFasterRCNN object detector.
    """
    import torch

    from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN

    # Define object detector
    object_detector = PyTorchFasterRCNN(
        input_shape=(3, 416, 416),
        clip_values=(0, 1),
        channels_first=True,
        attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
    )
    params = [p for p in object_detector.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01)

    object_detector.set_params(optimizer=optimizer)

    x_test, y_test = get_pytorch_object_detector_data

    yield object_detector, x_test, y_test


@pytest.fixture()
def get_pytorch_yolo(get_pytorch_object_detector_data):
    """
    This class tests the PyTorchYolo object detector.
    """
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

    x_test, y_test = get_pytorch_object_detector_data

    yield object_detector, x_test, y_test


@pytest.fixture()
def get_pytorch_detr(get_pytorch_object_detector_data):
    """
    This class tests the PyTorchDetectionTransformer object detector.
    """
    import torch

    from art.estimators.object_detection.pytorch_detection_transformer import PyTorchDetectionTransformer

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    object_detector = PyTorchDetectionTransformer(
        input_shape=(3, 416, 416),
        clip_values=(0, 1),
        preprocessing=(mean, std),
        channels_first=True,
        attack_losses=("loss_ce", "loss_bbox", "loss_giou"),
    )
    params = [p for p in object_detector.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01)

    object_detector.set_params(optimizer=optimizer)

    x_test, y_test = get_pytorch_object_detector_data

    yield object_detector, x_test, y_test
