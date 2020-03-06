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
This module implements base class for all estimators in ART.
"""

from abc import ABC, abstractmethod

import numpy as np

from art.defences.preprocessor.preprocessor import Preprocessor
from art.defences.postprocessor.postprocessor import Postprocessor


class BaseEstimator(ABC):
    """
    Base class defining the minimum requirements of a ART estimator. An estimator of this type can be combined with
    black-box attacks.
    """

    estimator_params = ["model", "clip_values", "preprocessing_defences", "postprocessing_defences", "preprocessing"]

    def __init__(
        self,
        model=None,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
    ):
        """
        Initialize a `Estimator` object.

        :param model: The model
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
               used for data preprocessing. The first value will be subtracted from the input and the results will be
               divided by the second value.
        :type preprocessing: `tuple`
        """
        kwargs = {
            "model": model,
            "clip_values": clip_values,
            "preprocessing_defences": preprocessing_defences,
            "postprocessing_defences": postprocessing_defences,
            "preprocessing": preprocessing,
        }
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before setting them as attributes.

        :param kwargs: a dictionary of attributes
        :type kwargs: `dict`
        :return: `self`
        """
        for key, value in kwargs.items():
            if key in self.estimator_params:
                if hasattr(BaseEstimator, key) and isinstance(getattr(BaseEstimator, key), property):
                    setattr(self, "_" + key, value)
                else:
                    setattr(self, key, value)
            else:
                raise ValueError("Unexpected parameter {} found in kwargs.".format(key))

        if isinstance(self.preprocessing_defences, Preprocessor):
            self.preprocessing_defences = [self.preprocessing_defences]

        if isinstance(self.postprocessing_defences, Postprocessor):
            self.postprocessing_defences = [self.postprocessing_defences]

        if self.preprocessing is not None and len(self.preprocessing) != 2:
            raise ValueError(
                "`preprocessing` should be a tuple of 2 floats with the values to subtract and divide the model inputs."
            )

        return self

    @abstractmethod
    def predict(self, x, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Perform prediction of the classifier for input `x`.

        :param x: Samples in an ndarray.
        :type x: `np.ndarray`
        :return: Array of predictions by the model of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Fit the classifier using the training data `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target data.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of estimator-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    @property
    def input_shape(self):
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        :rtype: `tuple`
        """
        return self._input_shape

    @property
    def clip_values(self):
        """
        Return the clip values of the input samples.

        :return: Clip values (min, max).
        :rtype: `tuple`
        """
        return self._clip_values

    def _apply_preprocessing(self, x, y, fit):
        """
        Apply all defences and preprocessing operations on the inputs `(x, y)`. This function has to be applied to all
        raw inputs (x, y) provided to the estimator.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param y: Target values (class labels), where first dimension is the number of samples.
        :type y: `np.ndarray` or `None`
        :param fit: `True` if the defences are applied during training.
        :type fit: `bool`
        :return: Value of the data after applying the defences.
        :rtype: `np.ndarray`
        """
        # y = check_and_transform_label_format(y, self.nb_classes)
        x_preprocessed, y_preprocessed = self._apply_preprocessing_defences(x, y, fit=fit)
        x_preprocessed = self._apply_preprocessing_standardisation(x_preprocessed)
        return x_preprocessed, y_preprocessed

    def _apply_preprocessing_defences(self, x, y, fit=False):
        """
        Apply all preprocessing defences of the estimator on the raw inputs `(x, y)`. This function is intended to
        only be called from function `_apply_preprocessing`.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param y: Target values (class labels), where first dimension is the number of samples.
        :type y: `np.ndarray`
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :type fit: `bool`
        :return: Arrays for `x` and `y` after applying the defences.
        :rtype: `np.ndarray`
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

        :param x: Input data, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :return: Array for `x` with the standardized data.
        :rtype: `np.ndarray`
        :raises: `TypeError`
        """
        if x.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            raise TypeError(
                "The data type of input data `x` is {} and cannot represent negative values. Consider "
                "changing the data type of the input data `x` to a type that supports negative values e.g. "
                "np.float32.".format(x.dtype)
            )

        if self.preprocessing is not None:
            sub, div = self.preprocessing
            sub = np.asarray(sub, dtype=x.dtype)
            div = np.asarray(div, dtype=x.dtype)

            res = x - sub
            res = res / div

        else:
            res = x

        return res

    def _apply_postprocessing(self, preds, fit):
        """
        Apply all postprocessing defences on model output.

        :param preds: model output to be post-processed.
        :type preds: `np.ndarray`
        :param fit: `True` if the defences are applied during training.
        :type fit: `bool`
        :return: Post-processed model output.
        :rtype: `np.ndarray`
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
        attributes = {(k[1:], v) if k[0] == "_" else (k, v) for (k, v) in self.__dict__.items()}
        attributes = ["{}={}".format(k, v) for (k, v) in attributes]
        repr_string = class_name + "(" + ", ".join(attributes) + ")"
        return repr_string


class LossGradientsMixin(ABC):
    """
    Base class defining additional classifier functionality for classifiers providing access to loss and class
    gradients. A classifier of this type can be combined with white-box attacks. This base class has to be mixed in with
    class `Classifier` and optionally class `ClassifierNeuralNetwork` to extend the minimum classifier functionality.
    """

    @abstractmethod
    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Input with shape as expected by the classifier's model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def _apply_preprocessing_gradient(self, x, gradients):
        """
        Apply the backward pass through all preprocessing operations to the gradients.

        Apply the backward pass through all preprocessing operations and defences on the inputs `(x, y)`. This function
        has to be applied to all gradients returned by the classifier.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param gradients: Input gradients.
        :type gradients: `np.ndarray`
        :return: Gradients after backward step through preprocessing operations and defences.
        :rtype: `np.ndarray`
        """
        gradients = self._apply_preprocessing_normalization_gradient(gradients)
        gradients = self._apply_preprocessing_defences_gradient(x, gradients)
        return gradients

    def _apply_preprocessing_defences_gradient(self, x, gradients, fit=False):
        """
        Apply the backward pass through the preprocessing defences.

        Apply the backward pass through all preprocessing defences of the classifier on the gradients. This function is
        intended to only be called from function `_apply_preprocessing_gradient`.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param gradients: Input gradient.
        :type gradients: `np.ndarray`
        :param fit: `True` if the gradient is computed during training.
        :return: Gradients after backward step through defences.
        :rtype: `np.ndarray`
        """
        if self.preprocessing_defences is not None:
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

        :param gradients: Input gradients.
        :type gradients: `np.ndarray`
        :return: Gradients after backward step through standardisation.
        :rtype: `np.ndarray
        """
        if self.preprocessing is not None:
            _, div = self.preprocessing
            div = np.asarray(div, dtype=gradients.dtype)
            res = gradients / div
        else:
            res = gradients

        return res


class NeuralNetworkMixin(ABC):
    """
    Base class defining additional classifier functionality required for neural network classifiers. This base class
    has to be mixed in with class `Classifier` to extend the minimum classifier functionality.
    """

    def __init__(self, channel_index=None, **kwargs):
        """
        Initialize a `ClassifierNeuralNetwork` object.

        :param channel_index: Index of the axis in input (feature) array `x` representing the color channels.
        :type channel_index: `int`
        """
        self._channel_index = channel_index
        super().__init__(**kwargs)

    @abstractmethod
    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction of the classifier for input `x`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param batch_size: The batch size used for evaluating the classifer's `model`.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: The batch size used for evaluating the classifer's `model`.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using `generator` yielding training batches as specified. Framework implementations can
        provide framework-specific versions of this function to speed-up computation.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        from art.data_generators import DataGenerator

        if not isinstance(generator, DataGenerator):
            raise ValueError(
                "Expected instance of `DataGenerator` for `fit_generator`, got %s instead." % str(type(generator))
            )

        for _ in range(nb_epochs):
            x, y = generator.get_batch()

            # Apply preprocessing and defences
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

            # Fit for current batch
            self.fit(x_preprocessed, y_preprocessed, nb_epochs=1, batch_size=len(x), **kwargs)

    @abstractmethod
    def get_activations(self, x, layer, batch_size):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abstractmethod
    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: `True` if the learning phase is training, `False` if learning phase is not training.
        :type train: `bool`
        """
        raise NotImplementedError

    @property
    def channel_index(self):
        """
        :return: Index of the axis in input data containing the color channels.
        :rtype: `int`
        """
        return self._channel_index

    @property
    def learning_phase(self):
        """
        Return the learning phase set by the user for the current classifier. Possible values are `True` for training,
        `False` for prediction and `None` if it has not been set through the library. In the latter case, the library
        does not do any explicit learning phase manipulation and the current value of the backend framework is used.
        If a value has been set by the user for this property, it will impact all following computations for
        model fitting, prediction and gradients.

        :return: Value of the learning phase.
        :rtype: `bool` or `None`
        """
        return self._learning_phase

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        return self._layer_names

    def __repr__(self):
        name = self.__class__.__name__

        attributes = {(k[1:], v) if k[0] == "_" else (k, v) for (k, v) in self.__dict__.items()}
        attrs = ["{}={}".format(k, v) for (k, v) in attributes]
        repr_ = name + "(" + ", ".join(attrs) + ")"

        return repr_


class DecisionTreeMixin(ABC):
    """
    Base class defining additional classifier functionality for decision-tree-based classifiers This base class has to
    be mixed in with class `Classifier` to extend the minimum classifier functionality.
    """

    @abstractmethod
    def get_trees(self):
        """
        Get the decision trees.

        :return: A list of decision trees.
        :rtype: `[Tree]`
        """
        raise NotImplementedError
