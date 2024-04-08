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
This module implements the task specific estimator for PyTorch YOLO v3 and v5 object detectors.

| Paper link: https://arxiv.org/abs/1804.02767
"""
import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchYolo(PyTorchObjectDetector):
    """
    This module implements the model- and task specific estimator for YOLO v3, v5 object detector models in PyTorch.

    | Paper link: https://arxiv.org/abs/1804.02767
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        input_shape: Tuple[int, ...] = (3, 416, 416),
        optimizer: Optional["torch.optim.Optimizer"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: bool = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: YOLO v3 or v5 model wrapped as demonstrated in examples/get_started_yolo.py.
                      The output of the model is `List[Dict[str, torch.Tensor]]`, one for each input image.
                      The fields of the Dict are as follows:

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
        """
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

    def _translate_labels(self, labels: List[Dict[str, "torch.Tensor"]]) -> "torch.Tensor":
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

    def _translate_predictions(self, predictions: "torch.Tensor") -> List[Dict[str, np.ndarray]]:
        """
        Translate object detection predictions from the model format (YOLO) to ART format (torchvision) and
        convert tensors to numpy arrays.

        :param predictions: Object detection labels in format xcycwh (YOLO).
        :return: Object detection labels in format x1y1x2y2 (torchvision).
        """
        import torch

        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        predictions_x1y1x2y2: List[Dict[str, np.ndarray]] = []

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
