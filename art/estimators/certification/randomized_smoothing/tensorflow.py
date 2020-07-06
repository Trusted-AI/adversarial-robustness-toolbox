# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements Randomized Smoothing applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1902.02918
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.estimators.certification.randomized_smoothing.randomized_smoothing import RandomizedSmoothingMixin

if TYPE_CHECKING:
    import tensorflow as tf

    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowV2RandomizedSmoothing(RandomizedSmoothingMixin, TensorFlowV2Classifier):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    def __init__(
        self,
        model,
        nb_classes: int,
        input_shape: Tuple[int, ...],
        loss_object: Optional["tf.Tensor"] = None,
        train_step: Optional[Callable] = None,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0, 1),
        sample_size: int = 32,
        scale: float = 0.1,
        alpha: float = 0.001,
    ):
        """
        Create a randomized smoothing classifier.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :type model: `function` or `callable class`
        :param nb_classes: the number of classes in the classification task.
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
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :param sample_size: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
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
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
        )

    def _predict_classifier(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        return TensorFlowV2Classifier.predict(self, x=x, batch_size=batch_size)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        return TensorFlowV2Classifier.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :key nb_epochs: Number of epochs to use for training
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        RandomizedSmoothingMixin.fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs)

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :param is_abstain: True if function will abstain from prediction and return 0s. Default: True
        :type is_abstain: `boolean`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        return RandomizedSmoothingMixin.predict(self, x, batch_size=128, **kwargs)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :param sampling: True if loss gradients should be determined with Monte Carlo sampling.
        :type sampling: `bool`
        :return: Array of gradients of the same shape as `x`.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        sampling = kwargs.get("sampling")

        if sampling:
            # Apply preprocessing
            x_preprocessed, _ = self._apply_preprocessing(x, y, fit=False)

            if tf.executing_eagerly():
                with tf.GradientTape() as tape:
                    inputs_t = tf.convert_to_tensor(x_preprocessed)
                    tape.watch(inputs_t)

                    inputs_repeat_t = tf.repeat(inputs_t, repeats=self.sample_size, axis=0)

                    noise = tf.random.normal(
                        shape=inputs_repeat_t.shape,
                        mean=0.0,
                        stddev=self.scale,
                        dtype=inputs_repeat_t.dtype,
                        seed=None,
                        name=None,
                    )

                    inputs_noise_t = inputs_repeat_t + noise
                    if self.clip_values is not None:
                        inputs_noise_t = tf.clip_by_value(
                            inputs_noise_t,
                            clip_value_min=self.clip_values[0],
                            clip_value_max=self.clip_values[1],
                            name=None,
                        )

                    model_outputs = self._model(inputs_noise_t)
                    softmax = tf.nn.softmax(model_outputs, axis=1, name=None)
                    average_softmax = tf.reduce_mean(
                        tf.reshape(softmax, shape=(-1, self.sample_size, model_outputs.shape[-1])), axis=1
                    )

                    loss = tf.reduce_mean(
                        tf.keras.losses.categorical_crossentropy(
                            y_true=y, y_pred=average_softmax, from_logits=False, label_smoothing=0
                        )
                    )

                gradients = tape.gradient(loss, inputs_t).numpy()
            else:
                raise ValueError("Expecting eager execution.")

            # Apply preprocessing gradients
            gradients = self._apply_preprocessing_gradient(x, gradients)

        else:
            gradients = TensorFlowV2Classifier.loss_gradient(self, x, y, **kwargs)

        return gradients

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        raise NotImplementedError
