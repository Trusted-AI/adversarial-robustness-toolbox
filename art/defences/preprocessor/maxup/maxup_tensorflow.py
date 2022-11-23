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
This module implements the Maxup data augmentation defence in TensorFlow.

| Paper link: https://arxiv.org/abs/2002.09024

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from art.defences.preprocessor.preprocessor import PreprocessorTensorFlowV2

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from art.estimators.classification import TensorFlowV2Classifier
    from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class MaxupTensorFlowV2(PreprocessorTensorFlowV2):
    """
    Implement the Maxup data augmentation defence approach in TensorFlow v2.

    | Paper link: https://arxiv.org/abs/2002.09024

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["num_classes", "alpha", "num_mix"]

    def __init__(
        self,
        estimator: "TensorFlowV2Classifier",
        augmentations: Union["Preprocessor", List["Preprocessor"]],
        num_trials: int = 1,
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of a Maxup data augmentation object.

        :param num_classes: The number of classes used for one-hot encoding.
        :param alpha: The hyperparameter for the mixing interpolation strength.
        :param num_mix: The number of samples to mix for k-way Mixup.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.estimator = estimator
        self.augmentations = estimator._set_preprocessing_defences(augmentations)
        self.num_trials = num_trials
        self._check_params()

    def forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Apply Mixup data augmentation to feature data `x` and labels `y`.

        :param x: Feature data to augment with shape `(batch_size, ...)`.
        :param y: Labels of `x` either one-hot or multi-hot encoded of shape `(nb_samples, nb_classes)`
                  or class indices of shape `(nb_samples,)`.
        :return: Data augmented sample. The returned labels will be probability vectors of shape
                 `(nb_samples, nb_classes)`.
        :raises `ValueError`: If no labels are provided.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if y is None:
            raise ValueError("Labels `y` cannot be None.")

        # extract the model
        model = self.estimator.model
        model.eval()

        # extract loss object and set to no reduction
        loss_object = self.estimator.loss_object
        prev_reduction = loss_object.reduction
        loss_object.reduction = tf.keras.losses.Reduction.NONE

        max_loss = 0
        x_max_loss = x
        y_max_loss = y

        for _ in range(self.num_trials):
            for augmentation in self.augmentations:
                # calculate the loss for the current augmentation
                x_aug, y_aug = augmentation(x, y)
                outputs = model(x_aug)
                loss = loss_object(outputs, y_aug)

                # one-hot encode if necessary
                if len(y_max_loss.shape) == 1 and len(y_aug.shape) == 2:
                    num_classes = y_aug.shape[1]
                    y_max_loss = tf.one_hot(y_max_loss, num_classes, on_value=1.0, off_value=0.0)
                elif len(y_max_loss.shape) == 2 and len(y_aug.shape) == 1:
                    num_classes = y_max_loss.shape[1]
                    y_aug = tf.one_hot(y_aug, num_classes, on_value=1.0, off_value=0.0)

                # select inputs and labels based on greater loss
                max_mask = tf.cast(max_loss > loss, x.dtype)
                max_loss = max_mask * max_loss + (1 - max_loss) * loss
                x_max_loss = max_mask * x_max_loss + (1 - max_loss) * x_aug
                y_max_loss = max_mask * y_max_loss + (1 - max_loss) * y_aug

        # restore original loss function reduction
        loss_object.reduction = prev_reduction

        return x_max_loss, y_max_loss

    def _check_params(self) -> None:
        if len(self.augmentations) == 0:
            raise ValueError("At least one augmentation must be provided.")

        if self.num_trials <= 0:
            raise ValueError("The number of trials must be positive.")
