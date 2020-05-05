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
This module implements gradient check functions for estimators
"""

import numpy as np


def loss_gradient_check(estimator, x, y):
    """
        Compute the gradient of the loss function w.r.t. `x` and identify points where the gradient is zero, nan, or inf

        :param estimator: The classifier to be analyzed.
        :type estimator: `art.estimators.estimator.BaseEstimator`
        :param x: Input with shape as expected by the classifier's model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of booleans with the shape (len(x), 3). If true means the gradient of the loss w.r.t. the
                 particular `x` was bad (zero, nan, inf)
        :rtype: `np.ndarray, np.ndarray`
        """
    assert len(x) == len(y), "x and y must be the same length"

    is_bad = []
    for i in range(len(x)):
        grad = estimator.loss_gradient([x[i]], [y[i]])
        is_bad.append([(np.min(grad) == 0 and np.max(grad) == 0), np.any(np.isnan(grad)), np.any(np.isinf(grad))])

    return np.array(is_bad, dtype=bool)
