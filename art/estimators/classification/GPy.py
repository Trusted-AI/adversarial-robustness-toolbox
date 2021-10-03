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
This module implements a wrapper class for GPy Gaussian Process classification models.
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np

from art.estimators.classification.classifier import ClassifierClassLossGradients
from art import config

if TYPE_CHECKING:
    # pylint: disable=C0412
    from GPy.models import GPClassification

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


# pylint: disable=C0103
class GPyGaussianProcessClassifier(ClassifierClassLossGradients):
    """
    Wrapper class for GPy Gaussian Process classification models.
    """

    def __init__(
        self,
        model: Optional["GPClassification"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Create a `Classifier` instance GPY Gaussian Process classification models.

        :param model: GPY Gaussian Process Classification model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """
        from GPy.models import GPClassification

        if not isinstance(model, GPClassification):  # pragma: no cover
            raise TypeError("Model must be of type GPy.models.GPClassification")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._nb_classes = 2  # always binary

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    # pylint: disable=W0221
    def class_gradient(  # type: ignore
        self, x: np.ndarray, label: Union[int, List[int], None] = None, eps: float = 0.0001, **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param eps: Fraction added to the diagonal elements of the input `x`.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        grads = np.zeros((np.shape(x_preprocessed)[0], 2, np.shape(x)[1]))
        for i in range(np.shape(x_preprocessed)[0]):
            # Get gradient for the two classes GPC can maximally have
            for i_c in range(2):
                ind = self.predict(x[i].reshape(1, -1))[0, i_c]
                sur = self.predict(
                    np.repeat(x_preprocessed[i].reshape(1, -1), np.shape(x_preprocessed)[1], 0)
                    + eps * np.eye(np.shape(x_preprocessed)[1])
                )[:, i_c]
                grads[i, i_c] = ((sur - ind) * eps).reshape(1, -1)

        grads = self._apply_preprocessing_gradient(x, grads)

        if label is not None:
            return grads[:, label, :].reshape(np.shape(x_preprocessed)[0], 1, np.shape(x_preprocessed)[1])

        return grads

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :return: Array of gradients of the same shape as `x`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y, fit=False)

        eps = 0.00001
        grads = np.zeros(np.shape(x))
        for i in range(np.shape(x)[0]):
            # 1.0 - to mimic loss, [0,np.argmax] to get right class
            ind = 1.0 - self.predict(x_preprocessed[i].reshape(1, -1))[0, np.argmax(y[i])]
            sur = (
                1.0
                - self.predict(
                    np.repeat(x_preprocessed[i].reshape(1, -1), np.shape(x_preprocessed)[1], 0)
                    + eps * np.eye(np.shape(x_preprocessed)[1])
                )[:, np.argmax(y[i])]
            )
            grads[i] = ((sur - ind) * eps).reshape(1, -1)

        grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    # pylint: disable=W0221
    def predict(self, x: np.ndarray, logits: bool = False, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param logits: `True` if the prediction should be done without squashing function.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Perform prediction
        out = np.zeros((np.shape(x_preprocessed)[0], 2))
        if logits:
            # output the non-squashed version
            out[:, 0] = self.model.predict_noiseless(x_preprocessed)[0].reshape(-1)
            out[:, 1] = -1.0 * out[:, 0]
        else:
            # output normal prediction, scale up to two values
            out[:, 0] = self.model.predict(x_preprocessed)[0].reshape(-1)
            out[:, 1] = 1.0 - out[:, 0]

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=out, fit=False)

        return predictions

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """
        Perform uncertainty prediction for a batch of inputs.

        :param x: Input samples.
        :return: Array of uncertainty predictions of shape `(nb_inputs)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Perform prediction
        out = self.model.predict_noiseless(x_preprocessed)[1]

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=out, fit=False)

        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data. Not used, as given to model in initialized earlier.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        """
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str] = None) -> None:  # pragma: no cover
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.model.save_model(full_path, save_data=False)
