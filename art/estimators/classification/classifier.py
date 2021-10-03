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
This module implements mixin abstract base classes defining properties for all classifiers in ART.
"""
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional, Union

import numpy as np

from art.estimators.estimator import (
    BaseEstimator,
    NeuralNetworkMixin,
    LossGradientsMixin,
    DecisionTreeMixin,
)


class InputFilter(ABCMeta):
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls.
    """

    def __init__(cls, name, bases, clsdict):  # pylint: disable=W0231,W0613
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

                if "x" in kwargs:  # pragma: no cover
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                else:
                    if not isinstance(args[0], np.ndarray):
                        lst[0] = np.array(args[0])

                if "y" in kwargs:  # pragma: no cover
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif has_y:  # pragma: no cover
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list_no_y = ["predict"]
        replacement_list_has_y = ["fit"]

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
    Mixin abstract base class defining functionality for classifiers.
    """

    estimator_params = ["nb_classes"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)  # type: ignore
        self._nb_classes: int = -1

    @property
    def nb_classes(self) -> int:
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        """
        if self._nb_classes < 2:  # pragma: no cover
            raise ValueError("nb_classes must be greater than or equal to 2.")
        return self._nb_classes  # type: ignore


class ClassGradientsMixin(ABC):
    """
    Mixin abstract base class defining classifiers providing access to class gradients. A classifier of this type can
    be combined with certain white-box attacks. This mixin abstract base class has to be mixed in with
    class `Classifier`.
    """

    @abstractmethod
    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Gradients of input features w.r.t. each class in the form `(batch_size, nb_classes, input_shape)` when
                 computing for all classes, otherwise shape becomes `(batch_size, 1, input_shape)` when `label`
                 parameter is specified.
        """
        raise NotImplementedError


class Classifier(ClassifierMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params


class ClassifierLossGradients(ClassifierMixin, LossGradientsMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params


class ClassifierClassLossGradients(ClassGradientsMixin, ClassifierMixin, LossGradientsMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params


class ClassifierNeuralNetwork(  # lgtm [py/conflicting-attributes]
    ClassGradientsMixin, ClassifierMixin, LossGradientsMixin, NeuralNetworkMixin, BaseEstimator, ABC
):
    """
    Typing variable definition.
    """

    estimator_params = (
        BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params + ClassifierMixin.estimator_params
    )

    @abstractmethod
    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework. This function is not supported for
        ensembles.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        raise NotImplementedError


class ClassifierDecisionTree(DecisionTreeMixin, ClassifierMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params
