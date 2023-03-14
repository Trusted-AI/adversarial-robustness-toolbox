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
This module implements the BadDet Regional Misclassification Attack (RMA) on object detectors.

| Paper link: https://arxiv.org/abs/2205.14497
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from art.attacks.attack import PoisoningAttackObjectDetector
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor

logger = logging.getLogger(__name__)


class BadDetRegionalMisclassificationAttack(PoisoningAttackObjectDetector):
    """
    Implementation of the BadDet Regional Misclassification Attack.

    | Paper link: https://arxiv.org/abs/2205.14497
    """

    attack_params = PoisoningAttackObjectDetector.attack_params + [
        "backdoor",
        "class_source",
        "class_target",
        "percent_poison",
        "channels_first",
        "verbose",
    ]
    _estimator_requirements = ()

    def __init__(
        self,
        backdoor: PoisoningAttackBackdoor,
        class_source: int = 0,
        class_target: int = 1,
        percent_poison: float = 0.3,
        channels_first: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Creates a new BadDet Regional Misclassification Attack

        :param backdoor: the backdoor chosen for this attack.
        :param class_source: The source class from which triggers were selected.
        :param class_target: The target label to which the poisoned model needs to misclassify.
        :param percent_poison: The ratio of samples to poison in the source class, with range [0, 1].
        :param channels_first: Set channels first or last.
        :param verbose: Show progress bars.
        """
        super().__init__()
        self.backdoor = backdoor
        self.class_source = class_source
        self.class_target = class_target
        self.percent_poison = percent_poison
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def poison(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: List[Dict[str, np.ndarray]],
        **kwargs,
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        """
        Generate poisoning examples by inserting the backdoor onto the input `x` and changing the classification
        for labels `y`.

        :param x: Sample images of shape `NCHW` or `NHWC`.
        :param y: True labels of type `List[Dict[np.ndarray]]`, one dictionary per input image. The keys and values
                  of the dictionary are:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
                  - scores [N]: the scores or each prediction.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        x_ndim = len(x.shape)

        if x_ndim != 4:
            raise ValueError("Unrecognized input dimension. BadDet RMA can only be applied to image data.")

        if self.channels_first:
            # NCHW --> NHWC
            x = np.transpose(x, (0, 2, 3, 1))

        x_poison = x.copy()
        y_poison: List[Dict[str, np.ndarray]] = []

        # copy labels and find indices of the source class
        source_indices = []
        for i, y_i in enumerate(y):
            target_dict = {k: v.copy() for k, v in y_i.items()}
            y_poison.append(target_dict)

            if self.class_source in y_i["labels"]:
                source_indices.append(i)

        # select indices of samples to poison
        num_poison = int(self.percent_poison * len(source_indices))
        selected_indices = np.random.choice(source_indices, num_poison, replace=False)

        for i in tqdm(selected_indices, desc="BadDet RMA iteration", disable=not self.verbose):
            image = x_poison[i]

            boxes = y_poison[i]["boxes"]
            labels = y_poison[i]["labels"]

            for j, (box, label) in enumerate(zip(boxes, labels)):
                if label == self.class_source:
                    # extract the bounding box from the image
                    x_1, y_1, x_2, y_2 = box.astype(int)
                    bounding_box = image[y_1:y_2, x_1:x_2, :]

                    # insert backdoor into the bounding box
                    # add an additional dimension to create a batch of size 1
                    poisoned_input, _ = self.backdoor.poison(bounding_box[np.newaxis], label)
                    image[y_1:y_2, x_1:x_2, :] = poisoned_input[0]

                    # change the source label to the target label
                    labels[j] = self.class_target

        if self.channels_first:
            # NHWC --> NCHW
            x_poison = np.transpose(x_poison, (0, 3, 1, 2))

        return x_poison, y_poison

    def _check_params(self) -> None:
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError("Backdoor must be of type PoisoningAttackBackdoor")
        if not 0 < self.percent_poison <= 1:
            raise ValueError("percent_poison must be between 0 and 1")
