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

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from art.config import ART_NUMPY_DTYPE
from art.estimators.object_detection import ObjectDetectorMixin, PyTorchObjectDetector, PyTorchFasterRCNN, PyTorchYolo
from art.estimators.pytorch import PyTorchEstimator
from art.utils import intersection_over_union, intersection_over_area, non_maximum_suppression

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchObjectSeeker(ObjectDetectorMixin, PyTorchEstimator):
    """
    Implementation of the ObjectSeeker certifiable robust defense applied to object detection models.
    The original implementation is https://github.com/inspire-group/ObjectSeeker

    | Paper link: https://arxiv.org/abs/2202.01811
    """

    estimator_params = PyTorchEstimator.estimator_params + [
        "input_shape",
        "optimizer",
        "detector_type",
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
        channels_first: Optional[bool] = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        detector_type: Literal["YOLO", "Faster-RCNN"] = "YOLO",
        num_lines: int = 3,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        prune_threshold: float = 0.5,
        epsilon: float = 0.1,
        device_type: str = "gpu",
        verbose: bool = False,
    ):
        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )

        self._input_shape = input_shape
        self._optimizer = optimizer
        self._attack_losses = attack_losses

        self.detector_type = detector_type
        self.num_lines = num_lines
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.prune_threshold = prune_threshold
        self.epsilon = epsilon
        self.verbose = verbose

        if detector_type == "YOLO":
            self.detector = PyTorchYolo(
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
        elif detector_type == "Faster-RCNN":
            self.detector = PyTorchFasterRCNN(
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
        else:
            self.detector = PyTorchObjectDetector(
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

    @property
    def native_label_is_pytorch_format(self) -> bool:
        """
        Return are the native labels in PyTorch format [x1, y1, x2, y2]?

        :return: Are the native labels in PyTorch format [x1, y1, x2, y2]?
        """
        return True

    @property
    def model(self) -> "torch.nn.Module":
        return self._model

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape

    @property
    def optimizer(self) -> Optional["torch.optim.Optimizer"]:
        """
        Return the optimizer.

        :return: The optimizer.
        """
        return self._optimizer

    @property
    def attack_losses(self) -> Tuple[str, ...]:
        """
        Return the combination of strings of the loss components.

        :return: The combination of strings of the loss components.
        """
        return self._attack_losses

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

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
        predictions = []

        for x_i in tqdm(x, desc="ObjectSeeker", disable=not self.verbose):
            base_preds = self.detector.predict(x_i[np.newaxis], batch_size=batch_size, **kwargs)[0]
            filtered_base_preds = non_maximum_suppression(
                base_preds, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold
            )

            masked_preds = self._masked_predictions(x_i, batch_size=batch_size, **kwargs)
            filtered_masked_preds = non_maximum_suppression(
                masked_preds, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold
            )

            pruned_preds = self._prune_boxes(filtered_masked_preds, filtered_base_preds)
            unionized_preds = self._unionize_clusters(pruned_preds)

            preds = {
                "boxes": np.concatenate([filtered_base_preds["boxes"], unionized_preds["boxes"]]),
                "labels": np.concatenate([filtered_base_preds["labels"], unionized_preds["labels"]]),
                "scores": np.concatenate([filtered_base_preds["scores"], unionized_preds["scores"]]),
            }

            predictions.append(preds)

        return predictions

    def _masked_predictions(self, x_i: np.ndarray, batch_size: int = 128, **kwargs) -> Dict[str, np.ndarray]:
        x_mask = np.repeat(x_i[np.newaxis, ...], self.num_lines * 4, axis=0)

        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
            x_mask = np.transpose(x_mask, (0, 3, 1, 2))

        idx = 0

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

        masked_preds = self.detector.predict(x=x_mask, batch_size=batch_size, **kwargs)
        filtered_preds = [
            non_maximum_suppression(
                pred, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold
            )
            for pred in masked_preds
        ]

        # Merge predictions
        boxes = np.concatenate([pred["boxes"] for pred in filtered_preds])
        labels = np.concatenate([pred["labels"] for pred in filtered_preds])
        scores = np.concatenate([pred["scores"] for pred in filtered_preds])

        merged_predictions = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

        return merged_predictions

    def _prune_boxes(
        self, masked_preds: Dict[str, np.ndarray], base_preds: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        masked_boxes = masked_preds["boxes"]
        masked_labels = masked_preds["labels"]
        masked_scores = masked_preds["scores"]

        base_boxes = base_preds["boxes"]
        base_labels = base_preds["labels"]

        keep_indices = []
        for idx, (masked_box, masked_label) in enumerate(zip(masked_boxes, masked_labels)):
            keep = True
            for (base_box, base_label) in zip(base_boxes, base_labels):
                if masked_label == base_label:
                    ioa = intersection_over_area(masked_box, base_box)
                    if ioa >= self.prune_threshold:
                        keep = False
                        break

            if keep:
                keep_indices.append(idx)

        pruned_preds = {
            "boxes": masked_boxes[keep_indices],
            "labels": masked_labels[keep_indices],
            "scores": masked_scores[keep_indices],
        }
        return pruned_preds

    def _unionize_clusters(self, masked_preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        boxes = masked_preds["boxes"]
        labels = masked_preds["labels"]
        scores = masked_preds["scores"]

        unionized_boxes = []
        unionized_labels = []
        unionized_scores = []

        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            selected_boxes = boxes[mask]
            selected_scores = scores[mask]

            areas = (selected_boxes[:, 2] - selected_boxes[:, 0]) * (selected_boxes[:, 3] - selected_boxes[:, 1])
            rb = np.minimum(selected_boxes[:, None, 2:], selected_boxes[:, 2:])
            lt = np.maximum(selected_boxes[:, None, :2], selected_boxes[:, :2])
            pairwise_intersection = np.prod(np.clip(rb - lt, 0, None), axis=2)
            pairwise_ioa = pairwise_intersection / (areas[:, None])
            distances = 1 - np.maximum(pairwise_ioa, pairwise_ioa.T)

            dbscan = DBSCAN(eps=0.1, min_samples=1, metric="precomputed")
            clusters = dbscan.fit_predict(distances)
            num_clusters = np.max(clusters) + 1

            for cluster in range(num_clusters):
                clustered_boxes = selected_boxes[clusters == cluster]

                clustered_box = [
                    np.min(clustered_boxes[:, 0]),
                    np.min(clustered_boxes[:, 1]),
                    np.max(clustered_boxes[:, 2]),
                    np.max(clustered_boxes[:, 3]),
                ]
                clustered_score = np.max(selected_scores)

                unionized_boxes.append(clustered_box)
                unionized_labels.append(label)
                unionized_scores.append(clustered_score)

        unionized_predictions = {
            "boxes": np.array(unionized_boxes),
            "labels": np.array(unionized_labels),
            "scores": np.array(unionized_scores),
        }
        return unionized_predictions

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
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
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
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
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
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
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
