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

# MIT License
#
# Copyright (c) 2022 Chong Xiang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the ObjectSeeker certifiably robust defense.

| Paper link: https://arxiv.org/abs/2202.01811
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from art.utils import intersection_over_area

logger = logging.getLogger(__name__)


class ObjectSeekerMixin(abc.ABC):
    """
    Implementation of the ObjectSeeker certifiable robust defense applied to object detection models.
    The original implementation is https://github.com/inspire-group/ObjectSeeker

    | Paper link: https://arxiv.org/abs/2202.01811
    """

    def __init__(
        self,
        *args,
        num_lines: int = 3,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        prune_threshold: float = 0.5,
        epsilon: float = 0.1,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Create an ObjectSeeker wrapper.

        :param num_lines: The number of divisions both vertically and horizontally to make masked predictions.
        :param confidence_threshold: The confidence threshold to discard bounding boxes.
        :param iou_threshold: The IoU threshold to discard overlapping bounding boxes.
        :param prune_threshold: The IoA threshold for pruning and duplicated bounding boxes.
        :param epsilon: The maximum distance between bounding boxes when merging using DBSCAN.
        :param verbose: Show progress bars.
        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.num_lines = num_lines
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.prune_threshold = prune_threshold
        self.epsilon = epsilon
        self.verbose = verbose

    @abc.abstractmethod
    def _image_dimensions(self) -> Tuple[int, int]:
        """
        Get the height and width of a sample input image.

        :return: Tuple containing the height and width of a sample input image.
        """
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    def _prune_boxes(
        self, masked_preds: Dict[str, np.ndarray], base_preds: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Remove bounding boxes from the masked predictions of a single image based on the IoA score with the boxes
        on the base unmasked predictions.

        :param masked_preds: The merged masked predictions of a single image.
        :param base_preds: The base unmasked predictions of a single image.
        :return: The filtered masked predictions with extraneous boxes removed.
        """
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
        """
        Cluster the bounding boxes for the pruned masked predictions.

        :param masked_preds: The merged masked predictions of a single image already pruned.
        :return: The clustered masked predictions with overlapping boxes merged.
        """
        boxes = masked_preds["boxes"]
        labels = masked_preds["labels"]
        scores = masked_preds["scores"]

        # Skip clustering if not enough bounding boxes
        if len(boxes) <= 1:
            return masked_preds

        unionized_boxes = []
        unionized_labels = []
        unionized_scores = []

        unique_labels = np.unique(labels)
        for label in unique_labels:
            # Select bounding boxes of same class
            mask = labels == label
            selected_boxes = boxes[mask]
            selected_scores = scores[mask]

            # Calculate pairwise distances between bounding boxes
            areas = (selected_boxes[:, 2] - selected_boxes[:, 0]) * (selected_boxes[:, 3] - selected_boxes[:, 1])
            top_left = np.maximum(selected_boxes[:, None, :2], selected_boxes[:, :2])
            bottom_right = np.minimum(selected_boxes[:, None, 2:], selected_boxes[:, 2:])
            pairwise_intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
            pairwise_ioa = pairwise_intersection / (areas[:, None])
            distances = 1 - np.maximum(pairwise_ioa, pairwise_ioa.T)

            # Cluster bounding boxes
            dbscan = DBSCAN(eps=self.epsilon, min_samples=1, metric="precomputed")
            clusters = dbscan.fit_predict(distances)
            num_clusters = np.max(clusters) + 1

            for cluster in range(num_clusters):
                # Merge all boxes in the cluster
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
            "boxes": np.asarray(unionized_boxes),
            "labels": np.asarray(unionized_labels),
            "scores": np.asarray(unionized_scores),
        }
        return unionized_predictions

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
            base_preds, masked_preds = self._masked_predictions(x_i, batch_size=batch_size, **kwargs)
            pruned_preds = self._prune_boxes(masked_preds, base_preds)
            unionized_preds = self._unionize_clusters(pruned_preds)

            preds = {
                "boxes": np.concatenate([base_preds["boxes"], unionized_preds["boxes"]]),
                "labels": np.concatenate([base_preds["labels"], unionized_preds["labels"]]),
                "scores": np.concatenate([base_preds["scores"], unionized_preds["scores"]]),
            }

            predictions.append(preds)

        return predictions

    def certify(
        self,
        x: np.ndarray,
        patch_size: float = 0.01,
        offset: float = 0.1,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Checks if there is certifiable IoA robustness for each predicted bounding box.

        :param x: Sample input with shape as expected by the model.
        :param patch_size: The size of the patch to check against.
        :param offset: The offset to distinguish between the far and near patches.
        :return: A list containing an array of bools for each bounding box per image indicating if the bounding
                 box is certified against the given patch.
        """
        height, width = self._image_dimensions()

        patch_size = np.sqrt(height * width * patch_size)
        height_offset = offset * height
        width_offset = offset * width

        # Get predictions
        predictions = self.predict(x, batch_size=batch_size)

        certifications: List[np.ndarray] = []

        for pred in tqdm(predictions, desc="ObjectSeeker", disable=not self.verbose):
            boxes = pred["boxes"]

            # Far patch
            far_patch_map = np.ones((len(boxes), height, width), dtype=bool)
            for i, box in enumerate(boxes):
                x_1 = int(max(0, box[1] - patch_size - height_offset))
                x_2 = int(min(box[3] + height_offset + 1, height))
                y_1 = int(max(0, box[0] - patch_size - width_offset))
                y_2 = int(min(box[2] + width_offset + 1, width))
                far_patch_map[i, x_1:x_2, y_1:y_2] = False
            far_vulnerable = np.any(far_patch_map, axis=(-2, -1))

            # Close patch
            close_patch_map = np.ones((len(boxes), height, width), dtype=bool)
            for i, box in enumerate(boxes):
                x_1 = int(max(0, box[1] - patch_size))
                x_2 = int(min(box[3] + 1, height))
                y_1 = int(max(0, box[0] - patch_size))
                y_2 = int(min(box[2] + 1, width))
                close_patch_map[i, x_1:x_2, y_1:y_2] = False
            close_vulnerable = np.any(close_patch_map, axis=(-2, -1))

            # Over patch
            close_patch_map = np.ones((len(boxes), height, width), dtype=bool)
            for i, box in enumerate(boxes):
                x_1 = int(max(0, box[1] - patch_size))
                x_2 = int(min(box[3] + 1, height))
                y_1 = int(max(0, box[0] - patch_size))
                y_2 = int(min(box[2] + 1, width))
                close_patch_map[i, x_1:x_2, y_1:y_2] = True
            over_vulnerable = np.any(close_patch_map, axis=(-2, -1))

            cert = np.logical_and.reduce((far_vulnerable, close_vulnerable, over_vulnerable))
            certifications.append(cert)

        return certifications
