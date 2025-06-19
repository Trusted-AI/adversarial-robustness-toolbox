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
"""
This module implements the task specific estimator for PyTorch YOLO v3, v5, v8+ object detectors.

| Paper link: https://arxiv.org/abs/1804.02767
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

if TYPE_CHECKING:

    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


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
            if 'v10' in name:
                self.model.criterion = E2EDetectLoss(model)
            else:
                self.model.criterion = v8DetectionLoss(model)
        except ImportError as e:
            raise ImportError("The 'ultralytics' package is required for YOLO v8+ models but not installed.") from e

    def forward(self, x, targets=None):
        if self.training:
            # batch_idx is used to identify which predictions/boxes relate to which image
            boxes = []
            labels = []
            indices = []
            for i, item in enumerate(targets):
               boxes.append(item['boxes']) 
               labels.append(item['labels'])
               indices = indices + ([i]*len(item['labels']))
            items = {'boxes': torch.concatenate(boxes) / x.shape[2],
                     'labels': torch.concatenate(labels).type(torch.float32),
                     'batch_idx': torch.tensor(indices)}
            items['bboxes'] = items.pop('boxes')
            items['cls'] = items.pop('labels')
            items['img'] = x
            
            loss, loss_components = self.model.loss(items)
            loss_components_dict = {"loss_total": loss.sum()}
            loss_components_dict['loss_box'] = loss_components[0]
            loss_components_dict['loss_cls'] = loss_components[1]
            loss_components_dict['loss_dfl'] = loss_components[2]
            return loss_components_dict
        else:
            preds = self.model(x)
            self.detection_predictor.model = self.model
            self.detection_predictor.batch = [x]
            preds = self.detection_predictor.postprocess(preds, x, x)
            # translate the preds to ART supported format
            items = []
            for pred in preds:
                items.append({'boxes': pred.boxes.xyxy,
                            'scores': pred.boxes.conf,
                            'labels': pred.boxes.cls.type(torch.int)}) 
            return items


class PyTorchYolo(PyTorchObjectDetector):
    """
    This module implements the model- and task specific estimator for YOLO v3, v5, v8+ object detector models in PyTorch.

    | Paper link: https://arxiv.org/abs/1804.02767
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        input_shape: tuple[int, ...] = (3, 416, 416),
        optimizer: "torch.optim.Optimizer" | None = None,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        channels_first: bool = True,
        preprocessing_defences: "Preprocessor" | list["Preprocessor"] | None = None,
        postprocessing_defences: "Postprocessor" | list["Postprocessor"] | None = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type: str = "gpu",
        is_yolov8: bool = False,
        model_name: str = "",
    ):
        """
        Initialization.

        :param model: YOLO v3, v5, or v8+ model wrapped as demonstrated in examples/get_started_yolo.py.
                      The output of the model is `list[dict[str, torch.Tensor]]`, one for each input image.
                      The fields of the dict are as follows:

                      - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                        0 <= y1 < y2 <= H.
                      - labels [N]: the labels for each image.
                      - scores [N]: the scores of each prediction.
        :param input_shape: The shape of one input sample.
        :param optimizer: The optimizer for training the classifier.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        :param is_yolov8: The flag to be used for marking the YOLOv8+ model.
        :param model_name: The name of the model (e.g., 'yolov8n', 'yolov10n') for determining loss function.
        """
        # Wrap the model with YoloWrapper if it's a YOLO v8+ model
        if is_yolov8:
            model = YoloWrapper(model, model_name)
        
        super().__init__(
            model=model,
            input_shape=input_shape,
            optimizer=optimizer,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            attack_losses=attack_losses,
            device_type=device_type,
            is_yolov8=is_yolov8,
        )

    def _translate_labels(self, labels: list[dict[str, "torch.Tensor"]]) -> "torch.Tensor":
        """
        Translate object detection labels from ART format (torchvision) to the model format (YOLO) and
        move tensors to GPU, if applicable.

        :param labels: Object detection labels in format x1y1x2y2 (torchvision).
        :return: Object detection labels in format xcycwh (YOLO).
        """
        import torch

        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        labels_xcycwh_list = []

        for i, label_dict in enumerate(labels):
            # create 2D tensor to encode labels and bounding boxes
            label_xcycwh = torch.zeros(len(label_dict["boxes"]), 6, device=self.device)
            label_xcycwh[:, 0] = i
            label_xcycwh[:, 1] = label_dict["labels"]
            label_xcycwh[:, 2:6] = label_dict["boxes"]

            # normalize bounding boxes to [0, 1]
            label_xcycwh[:, 2:6:2] /= width
            label_xcycwh[:, 3:6:2] /= height

            # convert from x1y1x2y2 to xcycwh
            label_xcycwh[:, 4] -= label_xcycwh[:, 2]
            label_xcycwh[:, 5] -= label_xcycwh[:, 3]
            label_xcycwh[:, 2] += label_xcycwh[:, 4] / 2
            label_xcycwh[:, 3] += label_xcycwh[:, 5] / 2
            labels_xcycwh_list.append(label_xcycwh)

        labels_xcycwh = torch.vstack(labels_xcycwh_list)
        return labels_xcycwh

    def _translate_predictions(self, predictions: "torch.Tensor") -> list[dict[str, np.ndarray]]:
        """
        Translate object detection predictions from the model format (YOLO) to ART format (torchvision) and
        convert tensors to numpy arrays.

        :param predictions: Object detection labels in format xcycwh (YOLO) or list of dicts (YOLO v8+).
        :return: Object detection labels in format x1y1x2y2 (torchvision).
        """
        import torch

        # Handle YOLO v8+ predictions (list of dicts)
        if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], dict):
            predictions_x1y1x2y2: list[dict[str, np.ndarray]] = []
            for pred in predictions:
                prediction = {}
                prediction["boxes"] = pred["boxes"].detach().cpu().numpy()
                prediction["labels"] = pred["labels"].detach().cpu().numpy()
                prediction["scores"] = pred["scores"].detach().cpu().numpy()
                predictions_x1y1x2y2.append(prediction)
            return predictions_x1y1x2y2

        # Handle traditional YOLO predictions (tensor format)
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        predictions_x1y1x2y2: list[dict[str, np.ndarray]] = []

        for pred in predictions:
            boxes = torch.vstack(
                [
                    torch.maximum((pred[:, 0] - pred[:, 2] / 2), torch.tensor(0, device=self.device)),
                    torch.maximum((pred[:, 1] - pred[:, 3] / 2), torch.tensor(0, device=self.device)),
                    torch.minimum((pred[:, 0] + pred[:, 2] / 2), torch.tensor(height, device=self.device)),
                    torch.minimum((pred[:, 1] + pred[:, 3] / 2), torch.tensor(width, device=self.device)),
                ]
            ).permute((1, 0))
            labels = torch.argmax(pred[:, 5:], dim=1)
            scores = pred[:, 4]

            pred_dict = {
                "boxes": boxes.detach().cpu().numpy(),
                "labels": labels.detach().cpu().numpy(),
                "scores": scores.detach().cpu().numpy(),
            }

            predictions_x1y1x2y2.append(pred_dict)

        return predictions_x1y1x2y2
