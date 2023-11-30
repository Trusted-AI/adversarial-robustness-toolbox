# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements the task specific estimator for DEtection TRansformer (DETR) in PyTorch.

| Paper link: https://arxiv.org/abs/2005.12872
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)


class PyTorchDetectionTransformer(PyTorchObjectDetector):
    """
    This class implements a model-specific object detector using DEtection TRansformer (DETR)
    and PyTorch following the input and output formats of torchvision.

    | Paper link: https://arxiv.org/abs/2005.12872
    """

    def __init__(
        self,
        model: Optional["torch.nn.Module"] = None,
        input_shape: Tuple[int, ...] = (3, 800, 800),
        optimizer: Optional["torch.optim.Optimizer"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: bool = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_ce",
            "loss_bbox",
            "loss_giou",
        ),
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: DETR model. The output of the model is `List[Dict[str, torch.Tensor]]`, one for each input
                      image. The fields of the Dict are as follows:

                      - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                        0 <= y1 < y2 <= H.
                      - labels [N]: the labels for each image.
                      - scores [N]: the scores of each prediction.
        :param input_shape: Tuple of the form `(height, width)` of ints representing input image height and width.
        :param optimizer: The optimizer for training the classifier.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_ce', 'loss_bbox', and
                              'loss_giou'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch
        from art.estimators.object_detection.detr import HungarianMatcher, SetCriterion, grad_enabled_forward

        if model is None:  # pragma: no cover
            model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)

        func_type = type(model.forward)
        model.forward = func_type(grad_enabled_forward, model)  # type: ignore

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
        )

        cost_class = 1.0
        cost_bbox = 5.0
        cost_giou = 2.0
        bbox_loss_coef = 5.0
        giou_loss_coef = 2.0
        eos_coef = 0.1
        num_classes = 91
        matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        losses = ["labels", "boxes", "cardinality"]

        self.weight_dict = {"loss_ce": 1, "loss_bbox": bbox_loss_coef, "loss_giou": giou_loss_coef}
        self.criterion = SetCriterion(
            num_classes, matcher=matcher, weight_dict=self.weight_dict, eos_coef=eos_coef, losses=losses
        )

    def _translate_labels(self, labels: List[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        """
        Translate object detection labels from ART format (torchvision) to the model format (DETR) and
        move tensors to GPU, if applicable.

        :param labels: Object detection labels in format x1y1x2y2 (torchvision).
        :return: Object detection labels in format xcycwh (DETR).
        """
        from art.estimators.object_detection.detr import revert_rescale_bboxes

        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        labels_translated: List[Dict[str, "torch.Tensor"]] = []

        for label_dict in labels:
            label_dict_translated = {}

            boxes = revert_rescale_bboxes(label_dict["boxes"], (height, width))
            label_dict_translated["boxes"] = boxes.to(self.device)

            label = label_dict["labels"]
            label_dict_translated["labels"] = label.to(self.device)

            if "scores" in label_dict:
                scores = label_dict["scores"]
                label_dict_translated["scores"] = scores.to(self.device)

            labels_translated.append(label_dict_translated)

        return labels_translated

    def _translate_predictions(self, predictions: Dict[str, "torch.Tensor"]) -> List[Dict[str, np.ndarray]]:
        """
        Translate object detection predictions from the model format (DETR) to ART format (torchvision) and
        convert tensors to numpy arrays.

        :param predictions: Object detection labels in format xcycwh (DETR).
        :return: Object detection labels in format x1y1x2y2 (torchvision).
        """
        from art.estimators.object_detection.detr import rescale_bboxes

        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        pred_boxes = predictions["pred_boxes"]
        pred_logits = predictions["pred_logits"]

        predictions_x1y1x2y2: List[Dict[str, np.ndarray]] = []

        for pred_box, pred_logit in zip(pred_boxes, pred_logits):
            boxes = rescale_bboxes(pred_box.detach().cpu(), (height, width)).numpy()
            labels = pred_logit.unsqueeze(0).softmax(-1)[0, :, :-1].max(dim=1)[1].detach().cpu().numpy()
            scores = pred_logit.unsqueeze(0).softmax(-1)[0, :, :-1].max(dim=1)[0].detach().cpu().numpy()

            pred_dict = {
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
            }

            predictions_x1y1x2y2.append(pred_dict)

        return predictions_x1y1x2y2
