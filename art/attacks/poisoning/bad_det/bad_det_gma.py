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
This module implements the BadDet Global Misclassification Attack (GMA) on object detectors.

| Paper link: https://arxiv.org/abs/2205.14497
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from art.attacks.attack import PoisoningAttackObjectDetector
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor

logger = logging.getLogger(__name__)


class BadDetGlobalMisclassificationAttack(PoisoningAttackObjectDetector):
    """
    Implementation of the BadDet Global Misclassification Attack.

    | Paper link: https://arxiv.org/abs/2205.14497
    """

    attack_params = PoisoningAttackObjectDetector.attack_params + [
        "backdoor",
        "class_target",
        "percent_poison",
        "channels_first",
        "verbose",
    ]
    _estimator_requirements = ()

    def __init__(
        self,
        backdoor: PoisoningAttackBackdoor,
        class_target: int = 1,
        percent_poison: float = 0.3,
        channels_first: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Creates a new BadDet Global Misclassification Attack

        :param backdoor: the backdoor chosen for this attack.
        :param class_target: The target label to which the poisoned model needs to misclassify.
        :param percent_poison: The ratio of samples to poison in the source class, with range [0, 1].
        :param channels_first: Set channels first or last.
        :param verbose: Show progress bars.
        """
        super().__init__()
        self.backdoor = backdoor
        self.class_target = class_target
        self.percent_poison = percent_poison
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def poison(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        y: List[Dict[str, np.ndarray]],
        **kwargs,
    ) -> Tuple[Union[np.ndarray, List[np.ndarray]], List[Dict[str, np.ndarray]]]:
        """
        Generate poisoning examples by inserting the backdoor onto the input `x` and changing the classification
        for labels `y`.

        :param x: Sample images of shape `NCHW` or `NHWC` or a list of sample images of any size.
        :param y: True labels of type `List[Dict[np.ndarray]]`, one dictionary per input image. The keys and values
                  of the dictionary are:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        if isinstance(x, np.ndarray):
            x_ndim = len(x.shape)
        else:
            x_ndim = len(x[0].shape) + 1

        if x_ndim != 4:
            raise ValueError("Unrecognized input dimension. BadDet GMA can only be applied to image data.")

        # copy images
        x_poison: Union[np.ndarray, List[np.ndarray]]
        if isinstance(x, np.ndarray):
            x_poison = x.copy()
        else:
            x_poison = [x_i.copy() for x_i in x]

        # copy labels
        y_poison: List[Dict[str, np.ndarray]] = []
        for y_i in y:
            target_dict = {k: v.copy() for k, v in y_i.items()}
            y_poison.append(target_dict)

        # select indices of samples to poison
        all_indices = np.arange(len(x))
        num_poison = int(self.percent_poison * len(all_indices))
        selected_indices = np.random.choice(all_indices, num_poison, replace=False)

        for i in tqdm(selected_indices, desc="BadDet GMA iteration", disable=not self.verbose):
            image = x_poison[i]
            labels = y_poison[i]["labels"]

            if self.channels_first:
                image = np.transpose(image, (1, 2, 0))

            # insert backdoor into the image
            # add an additional dimension to create a batch of size 1
            poisoned_input, _ = self.backdoor.poison(image[np.newaxis], labels)
            image = poisoned_input[0]

            # replace the original image with the poisoned image
            if self.channels_first:
                image = np.transpose(image, (2, 0, 1))
            x_poison[i] = image

            # change all labels to the target label
            y_poison[i]["labels"] = np.full(labels.shape, self.class_target)

        return x_poison, y_poison

    def _check_params(self) -> None:
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError("Backdoor must be of type PoisoningAttackBackdoor")
        if not 0 < self.percent_poison <= 1:
            raise ValueError("percent_poison must be between 0 and 1")
