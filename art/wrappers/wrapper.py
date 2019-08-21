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
Wrapper class for any classifier. Subclass of the ClassifierWrapper can override the behavior of key functions, such as
loss_gradient, to facilitate new attacks.
"""


class ClassifierWrapper:
    """
    Wrapper class for any classifier instance.
    """
    attack_params = ['classifier']

    def __init__(self, classifier):
        """
        Initialize a :class:`.ClassifierWrapper` object.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of an attack.
        :type classifier: :class:`.Classifier`
        """
        self.classifier = classifier

    def __getattr__(self, attr):
        """
        A generic grab-bag for the classifier instance. This makes the wrapped class look like a subclass.
        """
        return getattr(self.classifier, attr)

    def __setattr__(self, attr, value):
        """
        A generic grab-bag for the classifier instance. This makes the wrapped class look like a subclass.
        """
        if attr == 'classifier':
            object.__setattr__(self, attr, value)
        else:
            setattr(self.classifier, attr, value)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and pass them down to the underlying wrapped classifier instance.

        :param kwargs: A dictionary of attack-specific parameters.
        :type kwargs: `dict`
        :return: `True` when parsing was successful.
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True
