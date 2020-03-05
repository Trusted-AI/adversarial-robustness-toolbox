# MIT License
#
# Copyright (C) IBM Corporation 2018
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
This module implements abstract base classes defining to properties for all classifiers.
"""

from abc import ABC, ABCMeta, abstractmethod

import numpy as np

from art.utils import check_and_transform_label_format


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


class ClassifierMixin(ABC, metaclass=InputFilter):
    """
    Base class defining additional estimator functionality for classifiers.
    """

    #     def predict(self, x, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
    #         """
    #         Perform prediction of the classifier for input `x`.
    #
    #         :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
    #                   nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
    #
    #                   # The first dimension corresponds to the samples whereas all other dimensions
    #                   # correspond to the shape of a single sample. For example for feature vectors the shape of `x` is
    #                   # (nb_samples, nb_features), for images the shape is (nb_samples, nb_pixels_height, nb_pixels_width,
    #                   # nb_channels) in format NHWC or (nb_samples, nb_channels, nb_pixels_height, nb_pixels_width) in
    #                   # format NCHW. Any shape compatible with the model is accepted.
    #
    #         :type x: `np.ndarray`
    #         :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
    #         :rtype: `np.ndarray`
    #         """
    #         raise NotImplementedError
    #
    #     @abstractmethod
    #     def fit(self, x, y, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
    #         """
    #         Fit the classifier using the training data `(x, y)`.
    #
    #         :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
    #                   nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
    #         :type x: `np.ndarray`
    #         :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
    #                   (nb_samples,).
    #         :type y: `np.ndarray`
    #         :param kwargs: Dictionary of framework-specific arguments.
    #         :type kwargs: `dict`
    #         :return: `None`
    #         """
    #         raise NotImplementedError
    #

    #
    #
    #     def _apply_preprocessing(self, x, y, fit):
    #         """
    #         Apply all defences and preprocessing operations on the inputs `(x, y)`. This function has to be applied to all
    #         raw inputs (x, y) provided to the classifier.
    #
    #         :param x: Features, where first dimension is the number of samples.
    #         :type x: `np.ndarray`
    #         :param y: Target values (class labels), where first dimension is the number of samples.
    #         :type y: `np.ndarray` or `None`
    #         :param fit: `True` if the defences are applied during training.
    #         :type fit: `bool`
    #         :return: Value of the data after applying the defences.
    #         :rtype: `np.ndarray`
    #         """
    #         y = check_and_transform_label_format(y, self.nb_classes())
    #         x_preprocessed, y_preprocessed = self._apply_preprocessing_defences(x, y, fit=fit)
    #         x_preprocessed = self._apply_preprocessing_standardisation(x_preprocessed)
    #         return x_preprocessed, y_preprocessed
    #
    #     def _apply_preprocessing_defences(self, x, y, fit=False):
    #         """
    #         Apply all preprocessing defences of the classifier on the raw inputs `(x, y)`. This function is intended to
    #         only be called from function `_apply_preprocessing`.
    #
    #         :param x: Features, where first dimension is the number of samples.
    #         :type x: `np.ndarray`
    #         :param y: Target values (class labels), where first dimension is the number of samples.
    #         :type y: `np.ndarray`
    #         :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
    #                     predict operation.
    #         :type fit: `bool`
    #         :return: Arrays for `x` and `y` after applying the defences.
    #         :rtype: `np.ndarray`
    #         """
    #         if self.preprocessing_defences is not None:
    #             for defence in self.preprocessing_defences:
    #                 if fit:
    #                     if defence.apply_fit:
    #                         x, y = defence(x, y)
    #                 else:
    #                     if defence.apply_predict:
    #                         x, y = defence(x, y)
    #
    #         return x, y
    #
    #     def _apply_preprocessing_standardisation(self, x):
    #         """
    #         Apply standardisation to input data `x`.
    #
    #         :param x: Input data, where first dimension is the number of samples.
    #         :type x: `np.ndarray`
    #         :return: Array for `x` with the standardized data.
    #         :rtype: `np.ndarray`
    #         :raises: `TypeError`
    #         """
    #         if x.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
    #             raise TypeError(
    #                 "The data type of input data `x` is {} and cannot represent negative values. Consider "
    #                 "changing the data type of the input data `x` to a type that supports negative values e.g. "
    #                 "np.float32.".format(x.dtype)
    #             )
    #
    #         if self.preprocessing is not None:
    #             sub, div = self.preprocessing
    #             sub = np.asarray(sub, dtype=x.dtype)
    #             div = np.asarray(div, dtype=x.dtype)
    #
    #             res = x - sub
    #             res = res / div
    #
    #         else:
    #             res = x
    #
    #         return res
    #
    #     def _apply_postprocessing(self, preds, fit):
    #         """
    #         Apply all defences operations on model output.
    #
    #         :param preds: model output to be postprocessed.
    #         :type preds: `np.ndarray`
    #         :param fit: `True` if the defences are applied during training.
    #         :type fit: `bool`
    #         :return: Postprocessed model output.
    #         :rtype: `np.ndarray`
    #         """
    #         post_preds = preds.copy()
    #         if self.postprocessing_defences is not None:
    #             for defence in self.postprocessing_defences:
    #                 if fit:
    #                     if defence.apply_fit:
    #                         post_preds = defence(post_preds)
    #                 else:
    #                     if defence.apply_predict:
    #                         post_preds = defence(post_preds)
    #
    #         return post_preds
    #

    @property
    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes


class ClassGradientsMixin(ABC):
    """
    Base class defining additional classifier functionality for classifiers providing access to loss and class
    gradients. A classifier of this type can be combined with white-box attacks. This base class has to be mixed in with
    class `Classifier` and optionally class `ClassifierNeuralNetwork` to extend the minimum classifier functionality.
    """

    @abstractmethod
    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Input with shape as expected by the classifier's model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError
