# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements the regressors for scikit-learn models.
"""
import logging
import os
import pickle
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.scikitlearn import ScikitlearnEstimator
from art.estimators.regression import RegressorMixin
from art import config

if TYPE_CHECKING:
    # pylint: disable=C0412
    import sklearn

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class ScikitlearnRegressor(RegressorMixin, ScikitlearnEstimator):  # lgtm [py/missing-call-to-init]
    """
    Wrapper class for scikit-learn regression models.
    """

    estimator_params = ScikitlearnEstimator.estimator_params

    def __init__(
        self,
        model: "sklearn.base.BaseEstimator",
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Create a `Regressor` instance from a scikit-learn regressor model.

        :param model: scikit-learn regressor model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """
        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._input_shape = self._get_input_shape(model)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the regressor on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values.
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn` regressor and will be passed to this function as such.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)
        y_preprocessed = np.argmax(y_preprocessed, axis=1)

        self.model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self.model)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :return: Array of predictions.
        :raises `ValueError`: If the regressor does not have the method `predict`
        """
        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if callable(getattr(self.model, "predict", None)):
            y_pred = self.model.predict(x_preprocessed)
        else:
            raise ValueError("The provided model does not have the method `predict`.")

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=y_pred, fit=False)

        return predictions

    def save(self, filename: str, path: Optional[str] = None) -> None:
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

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self.model, file=file_pickle)

    def clone_for_refitting(self) -> "ScikitlearnRegressor":  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Create a copy of the classifier that can be refit from scratch.

        :return: new estimator
        """
        import sklearn  # lgtm [py/repeated-import]

        clone = type(self)(sklearn.base.clone(self.model))
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        return clone

    def reset(self) -> None:
        """
        Resets the weights of the classifier so that it can be refit from scratch.

        """
        # No need to do anything since scikitlearn models start from scratch each time fit() is called
        pass

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the MSE loss of the regressor for samples `x`.

        :param x: Input samples.
        :param y: Target values.
        :return: Loss values.
        """

        return (y - self.predict(x)) ** 2
