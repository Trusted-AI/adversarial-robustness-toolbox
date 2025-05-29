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
import pytest


@pytest.fixture()
def get_pytorch_detector_yolo():
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

    yield object_detector
