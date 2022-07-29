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
This module implements (De)Randomized Smoothing for Certifiable Defense against Patch Attacks

| Paper link: https://arxiv.org/abs/2002.10733
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.estimators.certification.derandomized_smoothing.derandomized_smoothing import DeRandomizedSmoothingMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowV2DeRandomizedSmoothing(DeRandomizedSmoothingMixin, TensorFlowV2Classifier):
    """
    Implementation of (De)Randomized Smoothing applied to classifier predictions as introduced
    in Levine et al. (2020).

    | Paper link: https://arxiv.org/abs/2002.10733
    """

    estimator_params = TensorFlowV2Classifier.estimator_params + [
        "ablation_type",
        "ablation_size",
        "threshold",
        "logits",
    ]

    def __init__(
        self,
        model,
        nb_classes: int,
        ablation_type: str,
        ablation_size: int,
        threshold: float,
        logits: bool,
        input_shape: Tuple[int, ...],
        loss_object: Optional["tf.Tensor"] = None,
        train_step: Optional[Callable] = None,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ):
        """
        Create a derandomized smoothing classifier.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :type model: `function` or `callable class`
        :param nb_classes: the number of classes in the classification task.
        :param ablation_type: The type of ablation to perform, must be either "column" or "block"
        :param ablation_size: The size of the data portion to retain after ablation. Will be a column of size N for
                              "column" ablation type or a NxN square for ablation of type "block"
        :param threshold: The minimum threshold to count a prediction.
        :param logits: if the model returns logits or normalized probabilities
        :param input_shape: Shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
            the model and computing gradients of the loss w.r.t. the input.
        :param train_step: A function that applies a gradient update to the trainable variables.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """
        super().__init__(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            loss_object=loss_object,
            train_step=train_step,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            ablation_type=ablation_type,
            ablation_size=ablation_size,
            threshold=threshold,
            logits=logits,
        )

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        import tensorflow as tf  # lgtm [py/repeated-import]

        outputs = TensorFlowV2Classifier.predict(
            self, x=x, batch_size=batch_size, training_mode=training_mode, **kwargs
        )
        if self.logits:
            outputs = tf.nn.softmax(outputs)
        return np.asarray(outputs >= self.threshold).astype(int)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        return TensorFlowV2Classifier.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Labels, one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter currently only supports
                       "scheduler" which is an optional function that will be called at the end of every
                       epoch to adjust the learning rate.
        """
        if self._train_step is None:  # pragma: no cover
            raise TypeError(
                "The training function `train_step` is required for fitting a model but it has not been " "defined."
            )

        scheduler = kwargs.get("scheduler")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        for epoch in tqdm(range(nb_epochs)):
            num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
            ind = np.arange(len(x_preprocessed))
            for m in range(num_batch):
                i_batch = np.copy(x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]])
                labels = y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]
                images = self.ablator.forward(i_batch)
                self._train_step(self.model, images, labels)

            if scheduler is not None:
                scheduler(epoch)

    def predict(
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:  # type: ignore
        """
        Perform prediction of the given classifier for a batch of inputs

        :param x: Input samples.
        :param batch_size: Batch size.
        :param training_mode: if to run the classifier in training mode
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        return DeRandomizedSmoothingMixin.predict(self, x, batch_size=batch_size, training_mode=training_mode, **kwargs)
