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
This module implements MACER applied to classifier predictions.

| Paper link: https://arxiv.org/abs/2001.02378
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

from tqdm.auto import trange
import numpy as np

from art.estimators.certification.randomized_smoothing.tensorflow import TensorFlowV2RandomizedSmoothing
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowV2MACER(TensorFlowV2RandomizedSmoothing):
    """
    Implementation of MACER training, as introduced in Zhai et al. (2020)

    | Paper link: https://arxiv.org/abs/2001.02378
    """

    estimator_params = TensorFlowV2RandomizedSmoothing.estimator_params + [
        "beta",
        "gamma",
        "lmbda",
        "gauss_num",
    ]

    def __init__(
        self,
        model,
        nb_classes: int,
        input_shape: Tuple[int, ...],
        loss_object: Optional["tf.Tensor"] = None,
        optimizer: Optional["tf.keras.optimizers.Optimizer"] = None,
        train_step: Optional[Callable] = None,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        sample_size: int = 32,
        scale: float = 0.1,
        alpha: float = 0.001,
        beta: float = 16.0,
        gamma: float = 8.0,
        lmbda: float = 12.0,
        gaussian_samples: int = 16,
    ) -> None:
        """
        Create a MACER classifier.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :type model: `function` or `callable class`
        :param nb_classes: the number of classes in the classification task.
        :param input_shape: Shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
               the model and computing gradients of the loss w.r.t. the input.
        :param optimizer: The optimizer used to train the classifier.
        :param train_step: A function that applies a gradient update to the trainable variables with signature
               `train_step(model, images, labels)`. This will override the default training loop that uses the
               provided `loss_object` and `optimizer` parameters. It is recommended to use the `@tf.function`
               decorator, if possible, for efficient training.
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
        :param sample_size: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
        :param beta: The inverse temperature.
        :param gamma: The hinge factor.
        :param lmbda: The trade-off factor.
        :param gaussian_samples: The number of gaussian samples per input.
        """
        super().__init__(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            loss_object=loss_object,
            optimizer=optimizer,
            train_step=train_step,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
        )
        self.beta = beta
        self.gamma = gamma
        self.lmbda = lmbda
        self.gaussian_samples = gaussian_samples

    def fit(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, verbose: bool = False, **kwargs
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Labels, one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param verbose: Display the training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. This parameter currently only supports
                       "scheduler" which is an optional function that will be called at the end of every
                       epoch to adjust the learning rate.
        """
        import tensorflow as tf

        if self._train_step is None:  # pragma: no cover
            if self._optimizer is None:  # pragma: no cover
                raise ValueError(
                    "An optimizer `optimizer` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )

            @tf.function
            def train_step(model, images, labels):
                with tf.GradientTape() as tape:
                    input_size = len(labels)

                    outputs = self.model(images, training=True)
                    outputs = tf.reshape(outputs, [input_size, self.gaussian_samples, self.nb_classes])

                    # Classification loss
                    outputs_softmax = tf.reduce_mean(tf.nn.softmax(outputs, axis=2), axis=1)
                    outputs_log_softmax = tf.math.log(outputs_softmax + 1e-10)
                    indices = tf.stack([np.arange(input_size), labels], axis=1)
                    nll_loss = tf.gather_nd(outputs_log_softmax, indices=indices)
                    classification_loss = -tf.reduce_sum(nll_loss)

                    # Robustness loss
                    beta_outputs = outputs * self.beta
                    beta_outputs_softmax = tf.reduce_mean(tf.nn.softmax(beta_outputs, axis=2), axis=1)
                    top2_score, top2_idx = tf.math.top_k(beta_outputs_softmax, k=2)
                    indices_correct = tf.cast(top2_idx[:, 0], labels.dtype) == labels
                    out = tf.boolean_mask(top2_score, indices_correct)
                    out0, out1 = out[:, 0], out[:, 1]
                    icdf_out1 = tf.math.erfinv(2 * out1 - 1) * np.sqrt(2)
                    icdf_out0 = tf.math.erfinv(2 * out0 - 1) * np.sqrt(2)
                    robustness_loss = icdf_out1 - icdf_out0
                    indices = (
                        ~tf.math.is_nan(robustness_loss)  # pylint: disable=E1130
                        & ~tf.math.is_inf(robustness_loss)  # pylint: disable=E1130
                        & (tf.abs(robustness_loss) <= self.gamma)
                    )
                    out0, out1 = out0[indices], out1[indices]
                    icdf_out1 = tf.math.erfinv(2 * out1 - 1) * np.sqrt(2)
                    icdf_out0 = tf.math.erfinv(2 * out0 - 1) * np.sqrt(2)
                    robustness_loss = icdf_out1 - icdf_out0 + self.gamma
                    robustness_loss = tf.reduce_sum(robustness_loss) * self.scale / 2

                    # Final objective function
                    loss = classification_loss + self.lmbda * robustness_loss
                    loss /= input_size

                gradients = tape.gradient(loss, model.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        else:
            train_step = self._train_step

        scheduler = kwargs.get("scheduler")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        train_ds = tf.data.Dataset.from_tensor_slices((x_preprocessed, y_preprocessed)).shuffle(10000).batch(batch_size)

        for epoch in trange(nb_epochs, disable=not verbose):
            for images, labels in train_ds:
                # Tile samples for Gaussian augmentation
                input_size = len(images)
                new_shape = [input_size * self.gaussian_samples]
                new_shape.extend(images[0].shape)
                images = tf.reshape(tf.tile(images, (1, 1, 1, self.gaussian_samples)), new_shape)

                # Add random noise for randomized smoothing
                noise = tf.random.normal(shape=images.shape, mean=0.0, stddev=self.scale)
                noisy_inputs = images + noise

                train_step(self.model, noisy_inputs, labels)

            if scheduler is not None:
                scheduler(epoch)
