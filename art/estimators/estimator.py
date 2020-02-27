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

from abc import ABC, ABCMeta, abstractmethod

import numpy as np

from art.defences.preprocessor.preprocessor import Preprocessor
from art.defences.postprocessor.postprocessor import Postprocessor


class InputFilter(ABCMeta):
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls.
    """

    def __init__(cls, name, bases, clsdict):
        """
        This function overrides any existing generate or extract methods with a new method that
        ensures the input is an ndarray. There is an assumption that the input object has implemented
        __array__ with np.array calls.
        """

        def make_replacement(fdict, func_name, has_y):
            """
            This function overrides creates replacement functions dynamically.
            """

            def replacement_function(self, *args, **kwargs):
                if len(args) > 0:
                    lst = list(args)

                if "x" in kwargs:
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                else:
                    if not isinstance(args[0], np.ndarray):
                        lst[0] = np.array(args[0])

                if "y" in kwargs:
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif has_y:
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list_no_y = ["predict", "get_activations", "class_gradient"]
        replacement_list_has_y = ["fit", "loss_gradient"]

        for item in replacement_list_no_y:
            if item in clsdict:
                new_function = make_replacement(clsdict, item, False)
                setattr(cls, item, new_function)
        for item in replacement_list_has_y:
            if item in clsdict:
                new_function = make_replacement(clsdict, item, True)
                setattr(cls, item, new_function)


class BaseEstimator(ABC, metaclass=InputFilter):
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
                setattr(self, key, value)
            else:
                raise ValueError("Unexpected parameter {} found in kwargs.".format(key))

        if isinstance(self.preprocessing_defences, Preprocessor):
            self.preprocessing_defences = [self.preprocessing_defences]

        if isinstance(self.postprocessing_defences, Postprocessor):
            self.postprocessing_defences = [self.postprocessing_defences]

        if self.preprocessing is not None and len(self.preprocessing) != 2:
            raise ValueError(
                "`preprocessing` should be a tuple of 2 floats with the values to subtract and divide"
                "the model inputs."
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
    @abstractmethod
    def input_shape(self):
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        :rtype: `tuple`
        """
        raise NotImplementedError

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
