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
from __future__ import absolute_import, division, print_function, unicode_literals


def check_is_fitted(model, parameters, error_msg=None):
    """
    Checks if the model is fitted by asserting the presence of the fitted parameters in the model.

    :param model: The model instance
    :param parameters: The name of the parameter or list of names of parameters that are fitted by the model
    :param error_msg: (string) Custom error message to be printed if the model is not fitted. Default message is
    'This model is not fitted yet. Call 'fit' with appropriate arguments before using this method.'
    :return: (bool) True if the model is fitted
    :raises: TypeError
    """
    if error_msg is None:
        error_msg = "This model is not fitted yet. Call 'fit' with appropriate arguments before using this method."

    if not hasattr(model, 'fit'):
        raise TypeError("%s cannot be fitted." % model)

    if not isinstance(parameters, (list, tuple)):
        parameters = [parameters]

    if not all([hasattr(model, param) for param in parameters]):
        return False

    return True
