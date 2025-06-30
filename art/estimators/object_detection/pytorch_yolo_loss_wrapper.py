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
"""
PyTorch-specific YOLO loss wrapper for ART for yolo versions 8 and above.
"""

import torch


class PyTorchYoloLossWrapper(torch.nn.Module):
    """Wrapper for YOLO v8+ models to handle loss dict format."""

    def __init__(self, model, name):
        super().__init__()
        self.model = model
        try:
            from ultralytics.models.yolo.detect import DetectionPredictor
            from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss

            self.detection_predictor = DetectionPredictor()
            self.model.args = self.detection_predictor.args
            if "v10" in name:
                self.model.criterion = E2EDetectLoss(model)
            else:
                self.model.criterion = v8DetectionLoss(model)
        except ImportError as e:
            raise ImportError("The 'ultralytics' package is required for YOLO v8+ models but not installed.") from e

    def forward(self, x, targets=None):
        if self.training:
            boxes = []
            labels = []
            indices = []
            for i, item in enumerate(targets):
                boxes.append(item["boxes"])
                labels.append(item["labels"])
                indices = indices + ([i] * len(item["labels"]))
            items = {
                "boxes": torch.cat(boxes) / x.shape[2],
                "labels": torch.cat(labels).type(torch.float32),
                "batch_idx": torch.tensor(indices),
            }
            items["bboxes"] = items.pop("boxes")
            items["cls"] = items.pop("labels")
            items["img"] = x
            loss, loss_components = self.model.loss(items)
            loss_components_dict = {"loss_total": loss.sum()}
            loss_components_dict["loss_box"] = loss_components[0].sum()
            loss_components_dict["loss_cls"] = loss_components[1].sum()
            loss_components_dict["loss_dfl"] = loss_components[2].sum()
            return loss_components_dict
        else:
            preds = self.model(x)
            self.detection_predictor.model = self.model
            self.detection_predictor.batch = [x]
            preds = self.detection_predictor.postprocess(preds, x, x)
            items = []
            for pred in preds:
                items.append(
                    {"boxes": pred.boxes.xyxy, "scores": pred.boxes.conf, "labels": pred.boxes.cls.type(torch.int)}
                )
            return items
