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
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.randomized_smoothing.randomized_smoothing import RandomizedSmoothingMixin
import art.estimators.certification.randomized_smoothing.smoothmix.train_smoothmix as train_smoothmix
from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art.attacks.evasion.iterative_method import BasicIterativeMethod
import torch

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchRandomizedSmoothing(RandomizedSmoothingMixin, PyTorchClassifier):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    estimator_params = PyTorchClassifier.estimator_params + [
        "sample_size", 
        "scale", 
        "alpha", 
        "num_noise_vec", 
        "train_multi_noise", 
        "attack_type",
        "epsilon",
        "num_steps",
        "warmup",
        "lbd",
        "gamma",
        "beta",
        "gauss_num",
        "eta",
        "mix_step",
        "maxnorm_s",
        "maxnorm"
    ]

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        scheduler: Optional["torch.optim.lr_scheduler"] = None,  # type: ignore
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
        sample_size: int = 32,
        scale: float = 0.1,
        alpha: float = 0.001,
        num_noise_vec: int = 1,
        train_multi_noise: bool = False,
        attack_type: str = "PGD",
        epsilon: float = 64.0,
        num_steps: int = 10,
        warmup: int = 1,
        lbd: float = 12.0,
        gamma: float = 8.0,
        beta: float = 16.0,
        gauss_num: int = 16,
        eta: float = 1.0,
        mix_step: int = 0,
        maxnorm_s: Optional[float] = None,
        maxnorm: Optional[float] = None,
        **kwargs
    ):
        """
        Create a randomized smoothing classifier.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param optimizer: The optimizer used to train the classifier.
        :param scheduler: The learning rate scheduler used to train the classifier.
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
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param sample_size: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
        :param num_noise_vec: Number of noise vectors
        :param train_multi_noise: Determines whether to use all the noise samples or not
        :param attack_type: The type of attack to use
        :param epsilon: Maximum perturbation that the attacker can introduce
        :param num_steps: Number of attack updates
        :param warmup: Warm-up strategy that is gradually increased for the first 10 epochs up to the original value of epsilon
        :param lbd: Weight of robustness loss in Macer
        :param gamma: Value to multiply the LR by
        :param beta: The inverse function temperature in Macer
        :param gauss_num: Number of gaussian samples per input
        :param eta: Hyperparameter to control the relative strength of the mixup loss in SmoothMix
        :param mix_step: Determines which sample to use for the clean side in SmoothMix
        :param maxnorm_s: initial value of alpha * mix_step
        :param maxnorm: initial value of alpha * mix_step for adversarial examples
        """
        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
            num_noise_vec=num_noise_vec,
            train_multi_noise=train_multi_noise,
            attack_type=attack_type,
            epsilon=epsilon,
            num_steps=num_steps,
            warmup=warmup,
            lbd=lbd,
            gamma=gamma,
            beta=beta,
            gauss_num=gauss_num,
            eta=eta,
            mix_step=mix_step,
            maxnorm_s=maxnorm_s,
            maxnorm=maxnorm,
            **kwargs
        )
        self.scheduler = scheduler

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        x = x.astype(ART_NUMPY_DTYPE)
        return PyTorchClassifier.predict(self, x=x, batch_size=batch_size, training_mode=training_mode, **kwargs)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        if "train_method" in kwargs:
            if kwargs.get("train_method") == "smoothmix":
                return train_smoothmix.fit_pytorch(self, x, y, batch_size, nb_epochs, **kwargs)
        else:
            g_a = GaussianAugmentation(sigma=self.scale, augmentation=False)
            x_rs, _ = g_a(x)
            x_rs = x_rs.astype(ART_NUMPY_DTYPE)
            return PyTorchClassifier.fit(self, x_rs, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        **kwargs
    ):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param batch_size: Batch size.
        :key nb_epochs: Number of epochs to use for training
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """

        # Set model mode
        self._model.train(mode=training_mode)

        RandomizedSmoothingMixin.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:  # type: ignore
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Input samples.
        :param batch_size: Batch size.
        :param is_abstain: True if function will abstain from prediction and return 0s. Default: True
        :type is_abstain: `boolean`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        return RandomizedSmoothingMixin.predict(self, x, batch_size=batch_size, training_mode=False, **kwargs)

    def loss_gradient(  # type: ignore
        self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param sampling: True if loss gradients should be determined with Monte Carlo sampling.
        :type sampling: `bool`
        :return: Array of gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        sampling = kwargs.get("sampling")

        if sampling:
            self._model.train(mode=training_mode)

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
            if self.clip_values is not None:
                inputs_noise_t.clamp(
                    torch.tensor(self.clip_values[0]),
                    torch.tensor(self.clip_values[1]),
                )

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
            gradients = inputs_t.grad.cpu().numpy().copy()  # type: ignore
            gradients = self._apply_preprocessing_gradient(x, gradients)
            assert gradients.shape == x.shape

        else:
            gradients = PyTorchClassifier.loss_gradient(self, x=x, y=y, training_mode=training_mode, **kwargs)

        return gradients

    def class_gradient(
        self, x: np.ndarray, label: Union[int, List[int], None] = None, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        raise NotImplementedError
    
    def _requires_grad_(self, model: torch.nn.Module, requires_grad: bool) -> None:
        """
        Enables gradients for the given model

        :param model: The model to enable gradients for
        :param requires_grad: Boolean to enable or disable gradients for all layers in the model
        """
        for param in model.parameters():
            param.requires_grad_(requires_grad)
    
    def _get_minibatches(self, x, y, num_batches):
        """
        Generate batches of the training data and target values

        :param X: Training data
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                    (nb_samples,).
        :param num_batches: The number of batches to generate
        """
        batch_size = len(x) // num_batches
        for i in range(num_batches):
            yield x[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]
