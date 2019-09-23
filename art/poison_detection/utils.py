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
Module providing convenience functions for poison detector algorithms
"""
from __future__ import absolute_import, division, print_function, unicode_literals


import logging
import numpy as np


logger = logging.getLogger(__name__)


def segment_by_class(data, classes, num_classes):
    """
    Returns segmented data according to specified features.

    :param data: data to be segmented
    :type data: `np.ndarray`
    :param classes: clesses used to segment data, e.g., segment according to predicted label or to `y_train` or other 
                    array of one hot encodings the same length as data
    :type classes: `np.ndarray`
    :param num_classes: how many features
    :type num_classes:
    :return: segmented data according to specified features.
    :rtype: `list`
    """
    by_class = [[] for _ in range(num_classes)]
    for indx, feature in enumerate(classes):
        if num_classes > 2:
            assigned = np.argmax(feature)
        else:
            assigned = int(feature)
        by_class[assigned].append(data[indx])

    return [np.asarray(i) for i in by_class]


def performance_diff(model1, model2, test_data, test_labels, perf_function='accuracy'):
    """
    Calculates the difference in performance between two models on the test_data with
    a performance function.

    Returns performance(model1) - performance(model2)

    :param model1: A trained ART classifier
    :type model1: `art.classifiers.classifier.Classifier`
    :param model2: A trained ART classifier
    :type model2: `art.classifiers.classifier.Classifier`
    :param test_data: The data to test both model's performance
    :type test_data: `np.ndarray`
    :param test_labels: The labels to the testing data
    :type test_labels: `np.ndarray`
    :param perf_function: The performance metric to be used
    :type perf_function: one of ['accuracy', 'f1'] or a callable function (true_labels, model_labels) -> float
    :return: the difference in performance
    :rtype: `float`
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    model1_labels = model1.predict(test_data)
    model2_labels = model2.predict(test_data)
    if perf_function == 'accuracy':
        model1_acc = accuracy_score(test_labels, model1_labels)
        model2_acc = accuracy_score(test_labels, model2_labels)
        return model1_acc - model2_acc
    elif perf_function == 'f1':
        model1_f1 = f1_score(test_labels, model1_labels)
        model2_f1 = f1_score(test_labels, model2_labels)
        return model1_f1 - model2_f1
    elif callable(perf_function):
        return perf_function(test_labels, model1_labels) - perf_function(test_labels, model2_labels)
    else:
        raise NotImplementedError("Performance function '{}' not supported".format(str(perf_function)))
