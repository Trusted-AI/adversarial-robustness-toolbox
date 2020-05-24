# MIT License
#
# Copyright (C) IBM Corporation 2020
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

import numpy as np

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.randomized_smoothing.randomized_smoothing import RandomizedSmoothingMixin

logger = logging.getLogger(__name__)


class PyTorchRandomizedSmoothing(RandomizedSmoothingMixin, PyTorchClassifier):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    def __init__(
        self,
        model,
        loss,
        input_shape,
        nb_classes,
        optimizer=None,
        channel_index=1,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1),
        device_type="gpu",
        sample_size=32,
        scale=0.1,
        alpha=0.001,
    ):
        """
        Create a randomized smoothing classifier.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :type model: `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :type loss: `torch.nn.modules.loss._Loss`
        :param input_shape: The shape of one input instance.
        :type input_shape: `tuple`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :type device_type: `string`
        :param sample_size: Number of samples for smoothing
        :type sample_size: `int`
        :param scale: Standard deviation of Gaussian noise added.
        :type scale: `float`
        :param alpha: The failure probability of smoothing
        :type alpha: `float`
        """
        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channel_index=channel_index,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
        )

    def _predict_classifier(self, x, batch_size):
        x = x.astype(np.float32)
        return PyTorchClassifier.predict(self, x=x, batch_size=batch_size)

    def _fit_classifier(self, x, y, batch_size, nb_epochs, **kwargs):
        x = x.astype(np.float32)
        return PyTorchClassifier.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param sampling: True if loss gradients should be determined with Monte Carlo sampling.
        :type sampling: `bool`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import torch

        sampling = kwargs.get("sampling")

        if sampling:
            # Apply preprocessing
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

            # Check label shape
            if self._reduce_labels:
                y_preprocessed = np.argmax(y_preprocessed, axis=1)

            # Convert the inputs to Tensors
            inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
            inputs_t.requires_grad = True

            # Convert the labels to Tensors
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)

            inputs_repeat_t = inputs_t.repeat_interleave(self.sample_size, 0)

            noise = torch.randn_like(inputs_repeat_t, device=self._device) * self.scale

            inputs_noise_t = inputs_repeat_t + noise

            inputs_noise_t.clamp(self.clip_values[0], self.clip_values[1])

            model_outputs = self._model(inputs_noise_t)[-1]

            softmax = torch.nn.functional.softmax(model_outputs, dim=1)

            average_softmax = (
                softmax.reshape(-1, self.sample_size, model_outputs.shape[-1]).mean(1, keepdim=True).squeeze(1)
            )

            log_softmax = torch.log(average_softmax.clamp(min=1e-20))

            loss = torch.nn.functional.nll_loss(log_softmax, labels_t)

            # Clean gradients
            self._model.zero_grad()

            # Compute gradients
            loss.backward()
            grads = inputs_t.grad.cpu().numpy().copy()
            grads = self._apply_preprocessing_gradient(x, grads)
            assert grads.shape == x.shape

        else:
            grads = PyTorchClassifier.loss_gradient(self, x, y, **kwargs)

        return grads

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError
