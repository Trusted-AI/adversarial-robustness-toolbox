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
This module implements SmoothAdv applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1906.04584
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

from tqdm.auto import trange
import numpy as np

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.estimators.certification.randomized_smoothing.tensorflow import TensorFlowV2RandomizedSmoothing
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowV2SmoothAdv(TensorFlowV2RandomizedSmoothing):
    """
    Implementation of SmoothAdv training, as introduced in Salman et al. (2019)

    | Paper link: https://arxiv.org/abs/1906.04584
    """

    estimator_params = TensorFlowV2RandomizedSmoothing.estimator_params + [
        "epsilon",
        "num_noise_vec",
        "num_steps",
        "warmup",
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
        epsilon: float = 1.0,
        num_noise_vec: int = 1,
        num_steps: int = 10,
        warmup: int = 1,
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
        :param epsilon: The maximum perturbation that can be induced.
        :param num_noise_vec: The number of noise vectors.
        :param num_steps: The number of attack updates.
        :param warmup: The warm-up strategy that is gradually increased up to the original value.
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
        self.epsilon = epsilon
        self.num_noise_vec = num_noise_vec
        self.num_steps = num_steps
        self.warmup = warmup

        classifier = TensorFlowV2Classifier(
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
        )
        self.attack = ProjectedGradientDescent(classifier, eps=self.epsilon, max_iter=1, verbose=False)

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
            if self._loss_object is None:  # pragma: no cover
                raise TypeError(
                    "A loss function `loss_object` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )
            if self._optimizer is None:  # pragma: no cover
                raise ValueError(
                    "An optimizer `optimizer` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )

            @tf.function
            def train_step(model, images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = self.loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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
            self.attack.norm = min(self.epsilon, (epoch + 1) * self.epsilon / self.warmup)

            for x_batch, y_batch in train_ds:
                mini_batch_size = len(x_batch) // self.num_noise_vec

                for mini_batch in range(self.num_noise_vec):
                    # Create mini batch
                    inputs = x_batch[mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size]
                    labels = y_batch[mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size]

                    # Tile samples for Gaussian augmentation
                    inputs = tf.reshape(tf.tile(inputs, (1, self.num_noise_vec, 1, 1)), x_batch.shape)
                    noise = tf.random.normal(inputs.shape, mean=0.0, stddev=self.scale)

                    original_inputs = inputs.numpy()
                    noise_for_attack = noise.numpy()
                    perturbation_delta = np.zeros_like(original_inputs)
                    for _ in range(self.num_steps):
                        perturbed_inputs = original_inputs + perturbation_delta
                        adv_ex = self.attack.generate(perturbed_inputs + noise_for_attack)
                        perturbation_delta = adv_ex - perturbed_inputs - noise_for_attack

                    # Add random noise for randomized smoothing
                    perturbed_inputs = original_inputs + perturbation_delta
                    noisy_inputs = tf.convert_to_tensor(perturbed_inputs) + noise

                    train_step(self.model, noisy_inputs, labels)

            if scheduler is not None:
                scheduler(epoch)
