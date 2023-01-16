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
This module implements the CutMix data augmentation defence.

| Paper link: https://arxiv.org/abs/1905.04899

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

logger = logging.getLogger(__name__)


class CutMixPyTorch(PreprocessorPyTorch):
    """
    Implement the CutMix data augmentation defence approach in PyTorch.

    | Paper link: https://arxiv.org/abs/1905.04899

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["num_classes", "alpha", "probability", "channels_first", "verbose"]

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        probability: float = 0.5,
        channels_first: bool = False,
        apply_fit: bool = True,
        apply_predict: bool = False,
        device_type: str = "gpu",
        verbose: bool = False,
    ) -> None:
        """
        Create an instance of a CutMix data augmentation object.

        :param num_classes: The number of classes used for one-hot encoding.
        :param alpha: The hyperparameter for sampling the combination ratio.
        :param probability: The probability of applying CutMix per sample.
        :param channels_first: Set channels first or last.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param verbose: Show progress bars.
        """
        super().__init__(
            device_type=device_type,
            is_fitted=True,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )
        self.num_classes = num_classes
        self.alpha = alpha
        self.probability = probability
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply CutMix data augmentation to sample `x`.

        :param x: Sample to augment with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`.
        :param y: Labels of `x` either one-hot or multi-hot encoded of shape `(nb_samples, nb_classes)`
                  or class indices of shape `(nb_samples,)`.
        :return: Data augmented sample. The returned labels will be probability vectors of shape
                 `(nb_samples, nb_classes)`.
        """
        import torch

        if y is None:
            raise ValueError("Labels `y` cannot be None.")

        # convert labels to one-hot encoding
        if len(y.shape) == 2:
            y_one_hot = y
        elif len(y.shape) == 1:
            y_one_hot = torch.nn.functional.one_hot(y, self.num_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised. "
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

        x_ndim = len(x.shape)

        # NCHW/NHWC/NFHWC --> NCFHW
        if x_ndim == 4:
            if self.channels_first:
                # NCHW --> NCFHW
                x_ncfhw = torch.unsqueeze(x, dim=2)
            else:
                # NHWC --> NCHW --> NCFHW
                x_nchw = torch.permute(x, (0, 3, 1, 2))
                x_ncfhw = torch.unsqueeze(x_nchw, dim=2)
        elif x_ndim == 5:
            if self.channels_first:
                # NCFHW
                x_ncfhw = x
            else:
                # NFHWC --> NCFHW
                x_ncfhw = torch.permute(x, (0, 4, 1, 2, 3))
        else:
            raise ValueError("Unrecognized input dimension. CutMix can only be applied to image and video data.")

        n, _, _, height, width = x_ncfhw.shape
        x_aug = x_ncfhw.clone()
        y_aug = y_one_hot.clone().float()

        # sample the combination ratio from the Beta distribution
        lmb = np.random.beta(self.alpha, self.alpha)
        cut_ratio = np.sqrt(1.0 - lmb)
        cut_height = int(height * cut_ratio)
        cut_width = int(width * cut_ratio)

        # randomly choose indices for samples to mix
        indices = torch.randperm(n, device=x.device)

        # generate a random bounding box per image
        for idx1, idx2 in enumerate(tqdm(indices, desc="CutMix", disable=not self.verbose)):
            prob = np.random.rand()
            if prob < self.probability:
                # uniform sampling
                center_x = torch.randint(0, height, (1,))
                center_y = torch.randint(0, width, (1,))
                bby1 = torch.clamp(center_y - cut_height // 2, 0, height)
                bbx1 = torch.clamp(center_x - cut_width // 2, 0, width)
                bby2 = torch.clamp(center_y + cut_height // 2, 0, height)
                bbx2 = torch.clamp(center_x + cut_width // 2, 0, width)

                # insert image bounding box
                x_aug[idx1, :, :, bbx1:bbx2, bby1:bby2] = x_ncfhw[idx2, :, :, bbx1:bbx2, bby1:bby2]  # type: ignore
                # mix labels
                y_aug[idx1] = lmb * y_aug[idx1] + (1.0 - lmb) * y_one_hot[idx2]  # type: ignore

        x_ncfhw = x_aug

        # NCHW/NHWC/NFHWC <-- NCFHW
        if x_ndim == 4:
            if self.channels_first:
                # NCHW <-- NCFHW
                x_aug = torch.squeeze(x_ncfhw, dim=2)
            else:
                # NHWC <-- NCHW <-- NCFHW
                x_nchw = torch.squeeze(x_ncfhw, dim=2)
                x_aug = torch.permute(x_nchw, (0, 2, 3, 1))
        elif x_ndim == 5:
            if self.channels_first:
                # NCFHW
                x_aug = x_ncfhw
            else:
                # NFHWC <-- NCFHW
                x_aug = torch.permute(x_ncfhw, (0, 2, 3, 4, 1))

        return x_aug, y_aug

    def _check_params(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("The number of classes must be positive")

        if self.alpha <= 0:
            raise ValueError("The combination ratio sampling parameter must be positive.")

        if self.probability < 0 or self.probability > 1:
            raise ValueError("The CutMix probability must be between 0 and 1.")
