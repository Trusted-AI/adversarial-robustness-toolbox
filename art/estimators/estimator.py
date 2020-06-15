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
This module implements abstract base and mixin classes for estimators in ART.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm import trange

from art.config import ART_NUMPY_DTYPE, CLIP_VALUES_TYPE, PREPROCESSING_TYPE
from art.defences.postprocessor.postprocessor import Postprocessor
from art.defences.preprocessor.preprocessor import Preprocessor
from art.utils import Deprecated, deprecated, deprecated_keyword_arg

if TYPE_CHECKING:
    from art.data_generators import DataGenerator
    from art.metrics.verification_decisions_trees import Tree


class BaseEstimator(ABC):
    """
    The abstract base class `BaseEstimator` defines the basic requirements of an estimator in ART. The BaseEstimator is
    is the highest abstraction of a machine learning model in ART.
    """

    estimator_params = [
        "model",
        "clip_values",
        "preprocessing_defences",
        "postprocessing_defences",
        "preprocessing",
    ]

    def __init__(
        self,
        model=None,
        clip_values: Optional[CLIP_VALUES_TYPE] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: PREPROCESSING_TYPE = (0, 1),
    ):
        """
        Initialize a `BaseEstimator` object.

        :param model: The model
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input and the results will be
               divided by the second value.
        """
        self._model = model
        self._clip_values = clip_values

        self.preprocessing_defences: Optional[List[Preprocessor]]
        if isinstance(preprocessing_defences, Preprocessor):
            self.preprocessing_defences = [preprocessing_defences]
        else:
            self.preprocessing_defences = preprocessing_defences

        self.postprocessing_defences: Optional[List[Postprocessor]]
        if isinstance(postprocessing_defences, Postprocessor):
            self.postprocessing_defences = [postprocessing_defences]
        else:
            self.postprocessing_defences = postprocessing_defences

        self.preprocessing = preprocessing
        self._check_params()

    def set_params(self, **kwargs) -> None:
        """
        Take a dictionary of parameters and apply checks before setting them as attributes.

        :param kwargs: A dictionary of attributes.
        """
        for key, value in kwargs.items():
            if key in self.estimator_params:
                if hasattr(BaseEstimator, key) and isinstance(getattr(BaseEstimator, key), property):
                    setattr(self, "_" + key, value)
                else:
                    setattr(self, key, value)
            else:
                raise ValueError("Unexpected parameter {} found in kwargs.".format(key))
        self._check_params()

    def get_params(self) -> Dict[str, Any]:
        """
        Get all parameters and their values of this estimator.

        :return: A dictionary of string parameter names to their value.
        """
        params = dict()
        for key in self.estimator_params:
            params[key] = getattr(self, key)
        return params

    def _check_params(self) -> None:
        if self._clip_values is not None:
            if len(self._clip_values) != 2:
                raise ValueError(
                    "`clip_values` should be a tuple of 2 floats or arrays containing the allowed data range."
                )
            if np.array(self._clip_values[0] >= self._clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

            if isinstance(self._clip_values, np.ndarray):
                self._clip_values = self._clip_values.astype(ART_NUMPY_DTYPE)
            else:
                self._clip_values = np.array(self._clip_values, dtype=ART_NUMPY_DTYPE)

        if isinstance(self.preprocessing_defences, list):
            for preproc_defence in self.preprocessing_defences:
                if not isinstance(preproc_defence, Preprocessor):
                    raise ValueError(
                        "All preprocessing defences have to be instance of "
                        "art.defences.preprocessor.preprocessor.Preprocessor."
                    )
        elif self.preprocessing_defences is None:
            pass
        else:
            raise ValueError(
                "All preprocessing defences have to be instance of "
                "art.defences.preprocessor.preprocessor.Preprocessor."
            )
        if isinstance(self.postprocessing_defences, list):
            for postproc_defence in self.postprocessing_defences:
                if not isinstance(postproc_defence, Postprocessor):
                    raise ValueError(
                        "All postprocessing defences have to be instance of "
                        "art.defences.postprocessor.postprocessor.Postprocessor."
                    )
        elif self.postprocessing_defences is None:
            pass
        else:
            raise ValueError(
                "All postprocessing defences have to be instance of "
                "art.defences.postprocessor.postprocessor.Postprocessor."
            )

        if self.preprocessing is not None and len(self.preprocessing) != 2:
            raise ValueError(
                "`preprocessing` should be a tuple of 2 floats with the values to subtract and divide the model inputs."
            )

    @abstractmethod
    def predict(self, x, *args, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Perform prediction of the estimator for input `x`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :return: Array of predictions by the model.
        :rtype: Format as produced by the `model`
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y, **kwargs) -> None:  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Fit the estimator using the training data `(x, y)`.

        :param x: Training data.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        """
        raise NotImplementedError

    @property
    def model(self):
        """
        Return the model.

        :return: The model.
        """
        return self._model

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def clip_values(self) -> Optional[CLIP_VALUES_TYPE]:
        """
        Return the clip values of the input samples.

        :return: Clip values (min, max).
        """
        return self._clip_values

    def _apply_preprocessing(self, x, y, fit: bool) -> Tuple[Any, Any]:
        """
        Apply all defences and preprocessing operations on the inputs `x` and `y`. This function has to be applied to
        all raw inputs `x` and `y` provided to the estimator.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model` or `None`
        :param fit: `True` if the defences are applied during training.
        :return: Tuple of `x` and `y` after applying the defences and standardisation.
        :rtype: Format as expected by the `model`
        """
        # y = check_and_transform_label_format(y, self.nb_classes)
        x_preprocessed, y_preprocessed = self._apply_preprocessing_defences(x, y, fit=fit)
        x_preprocessed = self._apply_preprocessing_standardisation(x_preprocessed)
        return x_preprocessed, y_preprocessed

    def _apply_preprocessing_defences(self, x, y, fit: bool = False) -> Tuple[Any, Any]:
        """
        Apply all preprocessing defences of the estimator on the raw inputs `x` and `y`. This function is should
        only be called from function `_apply_preprocessing`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :return: Tuple of `x` and `y` after applying the defences and standardisation.
        :rtype: Format as expected by the `model`
        """
        if self.preprocessing_defences is not None:
            for defence in self.preprocessing_defences:
                if fit:
                    if defence.apply_fit:
                        x, y = defence(x, y)
                else:
                    if defence.apply_predict:
                        x, y = defence(x, y)

        return x, y

    def _apply_preprocessing_standardisation(self, x):
        """
        Apply standardisation to input data `x`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :return: Standardized `x`.
        :rtype: Format as expected by the `model`
        :raises `TypeError`: If the input data type is unsigned.
        """
        if x.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            raise TypeError(
                "The data type of input data `x` is {} and cannot represent negative values. Consider "
                "changing the data type of the input data `x` to a type that supports negative values e.g. "
                "np.float32.".format(x.dtype)
            )

        if self.preprocessing is not None:
            sub, div = self.preprocessing

            if isinstance(x, np.ndarray):
                sub = np.asarray(sub, dtype=x.dtype)
                div = np.asarray(div, dtype=x.dtype)

            res = x - sub
            res = res / div

        else:
            res = x

        return res

    def _apply_postprocessing(self, preds, fit: bool) -> np.ndarray:
        """
        Apply all postprocessing defences on model predictions.

        :param preds: model output to be post-processed.
        :type preds: Format as expected by the `model`
        :param fit: `True` if the defences are applied during training.
        :return: Post-processed model predictions.
        """
        post_preds = preds.copy()
        if self.postprocessing_defences is not None:
            for defence in self.postprocessing_defences:
                if fit:
                    if defence.apply_fit:
                        post_preds = defence(post_preds)
                else:
                    if defence.apply_predict:
                        post_preds = defence(post_preds)

        return post_preds

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = {}
        for k, v in self.__dict__.items():
            k = k[1:] if k[0] == "_" else k
            attributes[k] = v
        attributes = ["{}={}".format(k, v) for k, v in attributes.items()]
        repr_string = class_name + "(" + ", ".join(attributes) + ")"
        return repr_string


class LossGradientsMixin(ABC):
    """
    Mixin abstract base class defining additional functionality for estimators providing loss gradients. An estimator
    of this type can be combined with white-box attacks. This mixin abstract base class has to be mixed in with
    class `BaseEstimator`.
    """

    @abstractmethod
    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :return: Loss gradients w.r.t. `x` in the same format as `x`.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    def _apply_preprocessing_gradient(self, x, gradients):
        """
        Apply the backward pass to the gradients through all normalization and preprocessing defences that have been
        applied to `x` and `y` in the forward pass. This function has to be applied to all gradients w.r.t. `x`
        calculated by the estimator.

        :param x: Features, where first dimension is the number of samples.
        :type x: Format as expected by the `model`
        :param gradients: Gradients before backward pass through normalization and preprocessing defences.
        :type gradients: Format as expected by the `model`
        :return: Gradients after backward pass through normalization and preprocessing defences.
        :rtype: Format as expected by the `model`
        """
        gradients = self._apply_preprocessing_normalization_gradient(gradients)
        gradients = self._apply_preprocessing_defences_gradient(x, gradients)
        return gradients

    def _apply_preprocessing_defences_gradient(self, x, gradients, fit=False):
        """
        Apply the backward pass to the gradients through all preprocessing defences that have been applied to `x`
        and `y` in the forward pass. This function is should only be called from function
        `_apply_preprocessing_gradient`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param gradients: Gradients before backward pass through preprocessing defences.
        :type gradients: Format as expected by the `model`
        :param fit: `True` if the gradients are computed during training.
        :return: Gradients after backward pass through preprocessing defences.
        :rtype: Format as expected by the `model`
        """
        if hasattr(self, "preprocessing_defences") and self.preprocessing_defences is not None:
            for defence in self.preprocessing_defences[::-1]:
                if fit:
                    if defence.apply_fit:
                        gradients = defence.estimate_gradient(x, gradients)
                else:
                    if defence.apply_predict:
                        gradients = defence.estimate_gradient(x, gradients)

        return gradients

    def _apply_preprocessing_normalization_gradient(self, gradients):
        """
        Apply the backward pass through standardisation of `x` to `gradients`.

        Apply the backward pass to the gradients through normalization that has been applied to `x` in the forward
        pass. This function is should only be called from function `_apply_preprocessing_gradient`.

        :param gradients: Gradients before backward pass through normalization.
        :type gradients: Format as expected by the `model`
        :return: Gradients after backward pass through normalization.
        """
        if hasattr(self, "preprocessing") and self.preprocessing is not None:
            _, div = self.preprocessing
            div = np.asarray(div, dtype=gradients.dtype)
            res = gradients / div
        else:
            res = gradients

        return res


class NeuralNetworkMixin(ABC):
    """
    Mixin abstract base class defining additional functionality required for neural network estimators. This base class
    has to be mixed in with class `BaseEstimator`.
    """

    @deprecated_keyword_arg("channel_index", end_version="1.5.0", replaced_by="channels_first")
    def __init__(self, channel_index=Deprecated, channels_first: Optional[bool] = None, **kwargs) -> None:
        """
        Initialize a neural network attributes.

        :param channel_index: Index of the axis in samples `x` representing the color channels.
        :type channel_index: `int`
        :param channels_first: Set channels first or last.
        """
        # Remove in 1.5.0
        if channel_index == 3:
            channels_first = False
        elif channel_index == 1:
            channels_first = True
        elif channel_index is not Deprecated:
            raise ValueError("Not a proper channel_index. Use channels_first.")

        self._channel_index = channel_index
        self._channels_first: Optional[bool] = channels_first
        super().__init__(**kwargs)

    @abstractmethod
    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :param nb_epochs: Number of training epochs.
        """
        raise NotImplementedError

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the estimator using a `generator` yielding training batches. Implementations can
        provide framework-specific versions of this function to speed-up computation.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :param nb_epochs: Number of training epochs.
        """
        from art.data_generators import DataGenerator

        if not isinstance(generator, DataGenerator):
            raise ValueError(
                "Expected instance of `DataGenerator` for `fit_generator`, got %s instead." % str(type(generator))
            )

        for i in range(nb_epochs):
            for _ in trange(
                int(generator.size / generator.batch_size), desc="Epoch %i/%i" % (i + 1, nb_epochs)  # type: ignore
            ):  # type: ignore
                x, y = generator.get_batch()

                # Apply preprocessing and defences
                x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)  # type: ignore

                # Fit for current batch
                self.fit(x_preprocessed, y_preprocessed, nb_epochs=1, batch_size=generator.batch_size, **kwargs)

    @abstractmethod
    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of a specific layer for samples `x` where `layer` is the index of the layer between 0 and
        `nb_layers - 1 or the name of the layer. The number of layers can be determined by counting the results
        returned by calling `layer_names`.

        :param x: Samples
        :param layer: Index or name of the layer.
        :param batch_size: Batch size.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        raise NotImplementedError

    @abstractmethod
    def set_learning_phase(self, train: bool) -> None:
        """
        Set the learning phase for the backend framework.

        :param train: `True` if the learning phase is training, otherwise `False`.
        """
        raise NotImplementedError

    @property  # type: ignore
    @deprecated(end_version="1.5.0", replaced_by="channels_first")
    def channel_index(self) -> Optional[int]:
        """
        :return: Index of the axis containing the color channels in the samples `x`.
        """
        return self._channel_index

    @property
    def channels_first(self) -> Optional[bool]:
        """
        :return: Boolean to indicate index of the color channels in the sample `x`.
        """
        return self._channels_first

    @property
    def learning_phase(self) -> Optional[bool]:
        """
        The learning phase set by the user. Possible values are `True` for training or `False` for prediction and
        `None` if it has not been set by the library. In the latter case, the library does not do any explicit learning
        phase manipulation and the current value of the backend framework is used. If a value has been set by the user
        for this property, it will impact all following computations for model fitting, prediction and gradients.

        :return: Learning phase.
        """
        return self._learning_phase  # type: ignore

    @property
    def layer_names(self) -> List[str]:
        """
        Return the names of the hidden layers in the model, if applicable.

        :return: The names of the hidden layers in the model, input and output layers are ignored.

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        return self._layer_names  # type: ignore

    def __repr__(self):
        name = self.__class__.__name__

        attributes = {}
        for k, v in self.__dict__.items():
            k = k[1:] if k[0] == "_" else k
            attributes[k] = v
        attrs = ["{}={}".format(k, v) for k, v in attributes.items()]
        repr_ = name + "(" + ", ".join(attrs) + ")"

        return repr_


class DecisionTreeMixin(ABC):
    """
    Mixin abstract base class defining additional functionality for decision-tree-based estimators. This mixin abstract
    base class has to be mixed in with class `BaseEstimator`.
    """

    @abstractmethod
    def get_trees(self) -> List["Tree"]:
        """
        Get the decision trees.

        :return: A list of decision trees.
        """
        raise NotImplementedError
