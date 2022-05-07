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
Module containing ART's exceptions.
"""
from typing import List, Tuple, Type, Union


class EstimatorError(TypeError):
    """
    Basic exception for errors raised by unexpected estimator types.
    """

    def __init__(self, this_class, class_expected_list: List[Union[Type, Tuple[Type]]], classifier_given) -> None:
        super().__init__()
        self.this_class = this_class
        self.class_expected_list = class_expected_list
        self.classifier_given = classifier_given

        classes_expected_message = ""
        for idx, class_expected in enumerate(class_expected_list):
            if idx != 0:
                classes_expected_message += " and "
            if isinstance(class_expected, type):
                classes_expected_message += f"{class_expected}"
            else:
                classes_expected_message += "("
                for or_idx, or_class in enumerate(class_expected):
                    if or_idx != 0:
                        classes_expected_message += " or "
                    classes_expected_message += f"{or_class}"
                classes_expected_message += ")"

        self.message = (
            f"{this_class.__name__} requires an estimator derived from {classes_expected_message}, "
            f"the provided classifier is an instance of {type(classifier_given)} "
            f"and is derived from {classifier_given.__class__.__bases__}."
        )

    def __str__(self) -> str:
        return self.message
