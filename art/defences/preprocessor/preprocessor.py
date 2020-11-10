# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the abstract base class for defences that pre-process input data.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import List, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np

from art import config

if TYPE_CHECKING:
    import torch
    import tensorflow as tf


class Preprocessor(abc.ABC):
    """
    Abstract base class for preprocessing defences.
    """

    params: List[str] = []

    def __init__(self) -> None:
        """
        Create a preprocessing object.
        """
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """
        Return the state of the preprocessing object.

        :return: `True` if the preprocessing model has been fitted (if this applies).
        """
        return self._is_fitted

    @property
    @abc.abstractmethod
    def apply_fit(self) -> bool:
        """
        Property of the defence indicating if it should be applied at training time.

        :return: `True` if the defence should be applied when fitting a model, `False` otherwise.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def apply_predict(self) -> bool:
        """
        Property of the defence indicating if it should be applied at test time.

        :return: `True` if the defence should be applied at prediction time, `False` otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Fit the parameters of the data preprocessor if it has any.

        :param x: Training set to fit the preprocessor.
        :param y: Labels for the training set.
        :param kwargs: Other parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Provide an estimate of the gradients of the defence for the backward pass. If the defence is not differentiable,
        this is an estimate of the gradient, most often replacing the computation performed by the defence with the
        identity function.

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :param grad: Gradient value so far.
        :return: The gradient (estimate) of the defence.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:
        pass

    def forward(self, x: Any, y: Any = None) -> Tuple[Any, Any]:
        """
        Perform data preprocessing and return preprocessed data.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError


class PreprocessorPyTorch(Preprocessor):
    """
    Abstract base class for preprocessing defences implemented in PyTorch that support efficient preprocessor-chaining.
    """

    @abc.abstractmethod
    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Perform data preprocessing in PyTorch and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        Provide a differentiable estimate of the forward function, so that autograd can calculate gradients
        of the defence for the backward pass. If the defence is differentiable, just call `self.forward()`.
        If the defence is not differentiable and a differentiable estimate is not available, replace with
        an identity function.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        import torch  # lgtm [py/repeated-import]

        x = torch.tensor(x, device=self._device)
        if y is not None:
            y = torch.tensor(y, device=self._device)

        with torch.no_grad():
            x, y = self.forward(x, y)

        result = x.cpu().numpy()
        if y is not None:
            y = y.cpu().numpy()
        return result, y

    # Backward compatibility.
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        import torch  # lgtm [py/repeated-import]

        def get_gradient(x, grad):
            x = torch.tensor(x, device=self._device, requires_grad=True)
            grad = torch.tensor(grad, device=self._device)

            x_prime = self.estimate_forward(x)
            x_prime.backward(grad)
            x_grad = x.grad.detach().cpu().numpy()

            if x_grad.shape != x.shape:
                raise ValueError("The input shape is {} while the gradient shape is {}".format(x.shape, x_grad.shape))

            return x_grad

        if x.shape == grad.shape:
            x_grad = get_gradient(x=x, grad=grad)
        else:
            # Special case for lass gradients
            x_grad = np.zeros_like(grad)
            for i in range(grad.shape[1]):
                x_grad[:, i, ...] = get_gradient(x=x, grad=grad[:, i, ...])

        return x_grad


class PreprocessorTensorFlowV2(Preprocessor):
    """
    Abstract base class for preprocessing defences implemented in TensorFlow v2 that support efficient
    preprocessor-chaining.
    """

    @abc.abstractmethod
    def forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Perform data preprocessing in TensorFlow v2 and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> "tf.Tensor":
        """
        Provide a differentiable estimate of the forward function, so that autograd can calculate gradients
        of the defence for the backward pass. If the defence is differentiable, just call `self.forward()`.
        If the defence is not differentiable and a differentiable estimate is not available, replace with
        an identity function.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        x = tf.convert_to_tensor(x)
        if y is not None:
            y = tf.convert_to_tensor(y)

        x, y = self.forward(x, y)

        result = x.numpy()
        if y is not None:
            y = y.numpy()
        return result, y

    # Backward compatibility.
    def estimate_gradient(self, x: "tf.Tensor", grad: "tf.Tensor") -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        def get_gradient(x, grad):
            with tf.GradientTape() as tape:
                x = tf.convert_to_tensor(x, dtype=config.ART_NUMPY_DTYPE)
                tape.watch(x)
                grad = tf.convert_to_tensor(grad, dtype=config.ART_NUMPY_DTYPE)

                x_prime = self.estimate_forward(x)

            x_grad = tape.gradient(target=x_prime, sources=x, output_gradients=grad)

            x_grad = x_grad.numpy()
            if x_grad.shape != x.shape:
                raise ValueError("The input shape is {} while the gradient shape is {}".format(x.shape, x_grad.shape))

            return x_grad

        if x.shape == grad.shape:
            x_grad = get_gradient(x=x, grad=grad)
        else:
            # Special case for lass gradients
            x_grad = np.zeros_like(grad)
            for i in range(grad.shape[1]):
                x_grad[:, i, ...] = get_gradient(x=x, grad=grad[:, i, ...])

        return x_grad
