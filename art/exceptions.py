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
Module containing ART's exceptions.
"""


class ClassifierError(TypeError):
    """
    Basic exception for errors raised by unexpected classifier types.
    """

    def __init__(self, this_class, class_expected_list, classifier_given):
        self.this_class = this_class
        self.class_expected_list = class_expected_list
        self.classifier_given = classifier_given

        classes_expected_message = ""
        for idx, class_expected in enumerate(class_expected_list):
            if idx == 0:
                classes_expected_message += "{0}".format(class_expected)
            else:
                classes_expected_message += " and {0}".format(class_expected)

        self.message = (
            "For {0} classifier must be an instance of {1}, "
            "the provided classifier is instance of {2}.".format(
                this_class.__name__, classes_expected_message, classifier_given
            )
        )
