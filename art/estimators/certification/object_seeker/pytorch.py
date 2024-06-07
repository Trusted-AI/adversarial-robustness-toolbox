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
This module implements the ObjectSeeker certifiably robust defense.

| Paper link: https://arxiv.org/abs/2202.01811
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.certification.object_seeker.object_seeker import ObjectSeekerMixin
from art.estimators.object_detection import (
    PyTorchObjectDetector,
    PyTorchFasterRCNN,
    PyTorchYolo,
    PyTorchDetectionTransformer,
)
from art.utils import non_maximum_suppression

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchObjectSeeker(ObjectSeekerMixin, PyTorchObjectDetector):
    """
    Implementation of the ObjectSeeker certifiable robust defense applied to object detection models.
    The original implementation is https://github.com/inspire-group/ObjectSeeker

    | Paper link: https://arxiv.org/abs/2202.01811
    """

    estimator_params = PyTorchObjectDetector.estimator_params + [
        "attack_losses",
        "num_lines",
        "confidence_threshold",
        "iou_threshold",
        "prune_threshold",
        "epsilon",
    ]

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
        detector_type: Literal["Faster-RCNN", "YOLO", "DETR"] = "YOLO",
        num_lines: int = 3,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        prune_threshold: float = 0.5,
        epsilon: float = 0.1,
        device_type: str = "gpu",
        verbose: bool = False,
    ):
        """
        Create an ObjectSeeker classifier.

        :param model: Object detection model. The output of the model is `List[Dict[str, torch.Tensor]]`,
                      one for each input image. The fields of the Dict are as follows:

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
        :param detector_type: The type of object detector being used: 'Faster-RCNN' | 'YOLO' | 'DETR'
        :param num_lines: The number of divisions both vertically and horizontally to make masked predictions.
        :param confidence_threshold: The confidence threshold to discard bounding boxes.
        :param iou_threshold: The IoU threshold to discard overlapping bounding boxes.
        :param prune_threshold: The IoA threshold for pruning and duplicated bounding boxes.
        :param epsilon: The maximum distance between bounding boxes when merging using DBSCAN.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        :param verbose: Show progress bars.
        """
        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
            num_lines=num_lines,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            prune_threshold=prune_threshold,
            epsilon=epsilon,
            verbose=verbose,
        )

        self._input_shape = input_shape
        self._channels_first = channels_first
        self._optimizer = optimizer
        self._attack_losses = attack_losses
        self.detector_type = detector_type

        detector_ctor: type
        if detector_type == "Faster-RCNN":
            detector_ctor = PyTorchFasterRCNN
        elif detector_type == "YOLO":
            detector_ctor = PyTorchYolo
        elif detector_type == "DETR":
            detector_ctor = PyTorchDetectionTransformer
        else:
            detector_ctor = PyTorchObjectDetector

        self.detector = detector_ctor(
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

    def _image_dimensions(self) -> Tuple[int, int]:
        """
        Return the height and width of a sample input image.

        :return: Tuple containing the height and width of a sample input image.
        """
        if self.channels_first:
            _, height, width = self.input_shape
        else:
            height, width, _ = self.input_shape

        return height, width

    def _masked_predictions(
        self, x_i: np.ndarray, batch_size: int = 128, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Create masked copies of the image for each of lines following the ObjectSeeker algorithm. Then creates
        predictions on the base unmasked image and each of the masked image.

        :param x_i: A single image of shape CHW or HWC.
        :batch_size: Batch size.
        :return: Predictions for the base unmasked image and merged predictions for the masked image.
        """
        x_mask = np.repeat(x_i[np.newaxis], self.num_lines * 4 + 1, axis=0)

        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
            x_mask = np.transpose(x_mask, (0, 3, 1, 2))

        idx = 1

        # Left masks
        for k in range(1, self.num_lines + 1):
            boundary = int(width / (self.num_lines + 1) * k)
            x_mask[idx, :, :, :boundary] = 0
            idx += 1

        # Right masks
        for k in range(1, self.num_lines + 1):
            boundary = width - int(width / (self.num_lines + 1) * k)
            x_mask[idx, :, :, boundary:] = 0
            idx += 1

        # Top masks
        for k in range(1, self.num_lines + 1):
            boundary = int(height / (self.num_lines + 1) * k)
            x_mask[idx, :, :boundary, :] = 0
            idx += 1

        # Bottom masks
        for k in range(1, self.num_lines + 1):
            boundary = height - int(height / (self.num_lines + 1) * k)
            x_mask[idx, :, boundary:, :] = 0
            idx += 1

        if not self.channels_first:
            x_mask = np.transpose(x_mask, (0, 2, 3, 1))

        predictions = self.detector.predict(x=x_mask, batch_size=batch_size, **kwargs)
        filtered_predictions = [
            non_maximum_suppression(
                pred, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold
            )
            for pred in predictions
        ]

        # Extract base predictions
        base_predictions = filtered_predictions[0]

        # Extract and merge masked predictions
        boxes = np.concatenate([pred["boxes"] for pred in filtered_predictions[1:]])
        labels = np.concatenate([pred["labels"] for pred in filtered_predictions[1:]])
        scores = np.concatenate([pred["scores"] for pred in filtered_predictions[1:]])
        merged_predictions = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }
        masked_predictions = non_maximum_suppression(
            merged_predictions, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold
        )

        return base_predictions, masked_predictions

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape NCHW or NHWC.
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict
                 are as follows:

                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image
                 - scores [N]: the scores or each prediction.
        """
        return ObjectSeekerMixin.predict(self, x=x, batch_size=batch_size, **kwargs)

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]],
        batch_size: int = 128,
        nb_epochs: int = 10,
        drop_last: bool = False,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
        """
        self.detector.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            nb_epochs=nb_epochs,
            drop_last=drop_last,
            scheduler=scheduler,
            **kwargs,
        )

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations
        :param batch_size: Size of batches.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        return self.detector.get_activations(
            x=x,
            layer=layer,
            batch_size=batch_size,
            framework=framework,
        )

    def loss_gradient(  # pylint: disable=W0613
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss gradients of the same shape as `x`.
        """
        return self.detector.loss_gradient(
            x=x,
            y=y,
            **kwargs,
        )

    def compute_losses(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all loss components.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Dictionary of loss components.
        """
        return self.detector.compute_losses(
            x=x,
            y=y,
        )

    def compute_loss(  # type: ignore
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss.
        """
        return self.detector.compute_loss(
            x=x,
            y=y,
            **kwargs,
        )
