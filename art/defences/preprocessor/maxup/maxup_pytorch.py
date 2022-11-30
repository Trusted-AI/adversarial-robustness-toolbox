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
This module implements the Maxup data augmentation defence in PyTorch.

| Paper link: https://arxiv.org/abs/2002.09024

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.estimators.classification import PyTorchClassifier
    from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class MaxupPyTorch(PreprocessorPyTorch):
    """
    Implement the Maxup data augmentation defence approach in PyTorch.

    | Paper link: https://arxiv.org/abs/2002.09024

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["estimator", "augmentations", "num_trials"]

    def __init__(
        self,
        estimator: "PyTorchClassifier",
        augmentations: Union["Preprocessor", List["Preprocessor"]],
        num_trials: int = 1,
        apply_fit: bool = False,
        apply_predict: bool = True,
        device_type: str = "gpu",
    ) -> None:
        """
        Create an instance of a Maxup data augmentation object.

        :param estimator: A trained estimator with a loss function.
        :param augmentations: The preprocessing data augmentation defence(s) to be applied.
        :param num_trials: The number of trials to attempt each augmentation.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__(
            device_type=device_type,
            is_fitted=True,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )
        self.estimator = estimator
        self.augmentations = estimator._set_preprocessing_defences(augmentations)
        self.num_trials = num_trials
        self._check_params()

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply Mixup data augmentation to feature data `x` and labels `y`.

        :param x: Feature data to augment with necessary shape for provided augmentations.
        :param y: Labels of `x` to augment with necessary shape for provided augmentations.
        :return: Data augmented sample. The returned labels will be shape of the provided augmentations.
        :raises `ValueError`: If no labels are provided.
        """
        import torch

        if y is None:
            raise ValueError("Labels `y` cannot be None.")

        # extract loss function and set to no reduction
        loss_fn = self.estimator.loss
        prev_reduction = loss_fn.reduction
        loss_fn.reduction = "none"

        max_loss = torch.zeros(len(x))
        x_max_loss = x
        y_max_loss = y

        for _ in range(self.num_trials):
            for augmentation in self.augmentations:
                # calculate the loss for the current augmentation
                x_aug, y_aug = augmentation(x.cpu().numpy(), y.cpu().numpy())
                preds = self.estimator.predict(x_aug)
                x_aug = torch.from_numpy(x_aug).to(self.device)
                y_aug = torch.from_numpy(y_aug).to(self.device)
                preds = torch.from_numpy(preds).to(self.device)
                loss = loss_fn(preds, y_aug)

                # one-hot encode if necessary
                if len(y_max_loss.shape) == 1 and len(y_aug.shape) == 2:
                    num_classes = y_aug.shape[1]
                    y_max_loss = torch.nn.functional.one_hot(y_max_loss, num_classes)
                elif len(y_max_loss.shape) == 2 and len(y_aug.shape) == 1:
                    num_classes = y_max_loss.shape[1]
                    y_aug = torch.nn.functional.one_hot(y_aug, num_classes)

                # select inputs and labels based on greater loss
                loss_mask = loss > max_loss
                x_mask = loss_mask.view(-1, *((1,) * (x_aug.ndim - 1))).expand_as(x_aug)
                y_mask = loss_mask.view(-1, *((1,) * (y_aug.ndim - 1))).expand_as(y_aug)

                max_loss = torch.where(loss_mask, loss, max_loss)
                x_max_loss = torch.where(x_mask, x_aug, x_max_loss)
                y_max_loss = torch.where(y_mask, y_aug, y_max_loss)

        # restore original loss function reduction
        loss_fn.reduction = prev_reduction

        return x_max_loss, y_max_loss

    def _check_params(self) -> None:
        if len(self.augmentations) == 0:
            raise ValueError("At least one augmentation must be provided.")

        if self.num_trials <= 0:
            raise ValueError("The number of trials must be positive.")
