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
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import pickle
import pytest
from tensorflow.keras.callbacks import LearningRateScheduler

from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing


# @pytest.mark.only_with_platform("kerastf")
# @pytest.mark.parametrize("loss_type", ["label", "function", "class"])
@pytest.mark.parametrize("loss_name",
                         ["categorical_crossentropy", "categorical_hinge", "sparse_categorical_crossentropy",
                          "kullback_leibler_divergence"])
def test_loss_functions2(get_image_classifier_list, get_default_mnist_subset, loss_name, supported_losses_proba,
                        supported_losses_logit, store_expected_values):
    # prediction and class_gradient should be independent of logits/probabilities and of loss function

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    import json
    with open("temp.json", "r") as f:
        expected_values = json.load(f)

    (y_test_pred_exp, class_gradient_prob_exp, class_grad_logit_exp, loss_grad_exp) = expected_values

    # loss_type_list = ["label", "function", "class"]
    loss_type_list = ["label", "function_losses", "function_backend"]

    for loss_type in loss_type_list:
        # if loss_name + "_" + loss_type not in supported_losses:
        #     tmp = ""

        # test_probabilities = True
        # if loss_type is "label" and loss_name not in ["categorical_crossentropy", "sparse_categorical_crossentropy"]:
        #     test_probabilities = False
        #     try:
        #         classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
        #                                                   from_logits=False)
        #     except Exception as e:
        #         tmp = ""

        if loss_name + "_" + loss_type in supported_losses_proba:
            classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
                                                      from_logits=False)

            y_test_pred_exp = np.argmax(classifier.predict(x=x_test_mnist), axis=1)
            np.testing.assert_array_equal(y_test_pred_exp, y_test_pred_exp)

            class_gradient = classifier.class_gradient(x_test_mnist, label=5)
            np.testing.assert_array_almost_equal(class_gradient[99, 0, 14, :, 0], class_gradient_prob_exp)

            loss_gradient_value = classifier.loss_gradient(x=x_test_mnist, y=y_test_mnist)
            np.testing.assert_array_almost_equal(loss_gradient_value[99, 14, :, 0], loss_grad_exp[loss_name])

        # testing with logits
        if loss_name + "_" + loss_type in supported_losses_logit:
        # if loss_type is not "label" and loss_name not in ["categorical_hinge", "kullback_leibler_divergence"]:
            classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
                                                      from_logits=True)

            y_test_pred_exp = np.argmax(classifier.predict(x=x_test_mnist), axis=1)
            np.testing.assert_array_equal(y_test_pred_exp, y_test_pred_exp)

            class_gradient = classifier.class_gradient(x_test_mnist, label=5)
            np.testing.assert_array_almost_equal(class_gradient[99, 0, 14, :, 0], class_grad_logit_exp)

            loss_gradient_value = classifier.loss_gradient(x=x_test_mnist, y=y_test_mnist)
            np.testing.assert_array_almost_equal(loss_gradient_value[99, 14, :, 0], loss_grad_exp[loss_name])


@pytest.mark.only_with_platform("kerastf")
@pytest.mark.parametrize("loss_type", ["label", "function", "class"])
@pytest.mark.parametrize("loss_name",
                         ["categorical_crossentropy", "categorical_hinge", "sparse_categorical_crossentropy",
                          "kullback_leibler_divergence"])
def test_loss_functions(get_image_classifier_list, get_default_mnist_subset, loss_type, loss_name,
                        store_expected_values, expected_values):
    # prediction and class_gradient should be independent of logits/probabilities and of loss function

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    (y_test_pred_exp, class_gradient_prob_exp, class_grad_logit_exp, loss_grad_exp) = expected_values

    # import json
    # with open("temp.json","w") as f:
    #     json.dump(expected_values, f, indent=4)

    test_probabilities = True
    if loss_type is "label" and loss_name not in ["categorical_crossentropy", "sparse_categorical_crossentropy"]:
        test_probabilities = False

    if test_probabilities:
        classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
                                                  from_logits=False)

        y_test_pred_exp = np.argmax(classifier.predict(x=x_test_mnist), axis=1)
        np.testing.assert_array_equal(y_test_pred_exp, y_test_pred_exp)

        class_gradient = classifier.class_gradient(x_test_mnist, label=5)
        np.testing.assert_array_almost_equal(class_gradient[99, 0, 14, :, 0], class_gradient_prob_exp)

        loss_gradient_value = classifier.loss_gradient(x=x_test_mnist, y=y_test_mnist)
        np.testing.assert_array_almost_equal(loss_gradient_value[99, 14, :, 0], loss_grad_exp[loss_name])

    # testing with logits
    if loss_type is not "label" and loss_name not in ["categorical_hinge", "kullback_leibler_divergence"]:
        classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
                                                  from_logits=True)

        y_test_pred_exp = np.argmax(classifier.predict(x=x_test_mnist), axis=1)
        np.testing.assert_array_equal(y_test_pred_exp, y_test_pred_exp)

        class_gradient = classifier.class_gradient(x_test_mnist, label=5)
        np.testing.assert_array_almost_equal(class_gradient[99, 0, 14, :, 0], class_grad_logit_exp)

        loss_gradient_value = classifier.loss_gradient(x=x_test_mnist, y=y_test_mnist)
        np.testing.assert_array_almost_equal(loss_gradient_value[99, 14, :, 0], loss_grad_exp[loss_name])


@pytest.mark.only_with_platform("kerastf")
def test_learning_phase(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    assert hasattr(classifier, "_learning_phase") is False
    classifier.set_learning_phase(False)
    assert classifier.learning_phase is False
    classifier.set_learning_phase(True)
    assert classifier.learning_phase
    assert hasattr(classifier, "_learning_phase")



