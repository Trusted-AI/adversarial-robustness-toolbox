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


def _run_tests(
        _class_gradient_probabilities_expected,
        _loss_gradient_expected,
        classifier,
        x_test_mnist,
        y_test_mnist
):

    y_test_pred_expected = np.asarray(
        [
            7,
            1,
            1,
            4,
            4,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            1,
            4,
            4,
            7,
            8,
            4,
            4,
            4,
            4,
            3,
            4,
            4,
            7,
            4,
            4,
            4,
            7,
            4,
            3,
            4,
            7,
            0,
            7,
            7,
            1,
            1,
            7,
            7,
            4,
            0,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            7,
            4,
            4,
            4,
            4,
            4,
            1,
            4,
            4,
            7,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            0,
            7,
            4,
            0,
            1,
            7,
            4,
            4,
            7,
            4,
            4,
            4,
            4,
            4,
            7,
            4,
            4,
            4,
            4,
            4,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ]
    )

    y_test_pred = np.argmax(classifier.predict(x=x_test_mnist), axis=1)
    np.testing.assert_array_equal(y_test_pred, y_test_pred_expected)

    class_gradient = classifier.class_gradient(x_test_mnist, label=5)
    np.testing.assert_array_almost_equal(
        class_gradient[99, 0, 14, :, 0], _class_gradient_probabilities_expected
    )

    loss_gradient = classifier.loss_gradient(x=x_test_mnist, y=y_test_mnist)
    np.testing.assert_array_almost_equal(loss_gradient[99, 14, :, 0], _loss_gradient_expected)


@pytest.mark.only_with_platform("kerastf")
@pytest.mark.parametrize("loss_type", ["label", "function", "class"])
def test_loss_function_categorical_hinge(get_image_classifier_list, get_default_mnist_subset, loss_type):
    # prediction and class_gradient should be independent of logits/probabilities and of loss function

    class_gradient_probabilities_expected = np.asarray(
        [
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            2.3582461e-03,
            4.8802234e-04,
            1.6699843e-03,
            -6.4777887e-05,
            -1.4215634e-03,
            -1.3359448e-04,
            2.0448549e-03,
            2.8171093e-04,
            1.9665064e-04,
            1.5335126e-03,
            1.7000455e-03,
            -2.0136381e-04,
            6.4588618e-04,
            2.0524357e-03,
            2.1990810e-03,
            8.3692279e-04,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )

    class_gradient_logits_expected = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.08147776,
            0.01847786,
            0.07045883,
            -0.00269106,
            -0.03189164,
            0.01643312,
            0.1185048,
            0.02166386,
            0.00905327,
            0.06592228,
            0.04471018,
            -0.02879605,
            0.04668707,
            0.06856851,
            0.06857751,
            0.00657996,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset



    # ================= #
    # categorical_hinge #
    # ================= #

    loss_name = "categorical_hinge"

    # loss_gradient should be the same for probabilities and logits but dependent on loss function

    loss_gradient_expected = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            -0.00644846,
            -0.00274792,
            -0.01334668,
            -0.01417109,
            0.03608133,
            0.00670766,
            0.02741555,
            -0.00938758,
            0.002172,
            0.01400783,
            -0.00224488,
            -0.01445661,
            0.01775403,
            0.00643058,
            0.00382644,
            -0.01077214,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # testing with probabilities
    # if loss_name is "categorical_hinge":

    if loss_name is "categorical_hinge" and loss_type is not "label":
        classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
                                                  from_logits=False)
        _run_tests(
            class_gradient_probabilities_expected,
            loss_gradient_expected,
            classifier,
            x_test_mnist,
            y_test_mnist
        )

    # ======================== #
    # categorical_crossentropy #
    # ======================== #

    loss_name = "categorical_crossentropy"

    # loss_gradient should be the same for probabilities and logits but dependent on loss function

    loss_gradient_expected = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            -0.09573442,
            -0.0089094,
            0.01402334,
            0.0258659,
            0.08960329,
            0.10324767,
            0.10624839,
            0.06578761,
            -0.00018638,
            -0.01345262,
            -0.08770822,
            -0.04990875,
            0.04288402,
            -0.06845165,
            -0.08588978,
            -0.08277036,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # testing with probabilities
    classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
                                              from_logits=False)
    _run_tests(
        class_gradient_probabilities_expected,
        loss_gradient_expected,
        classifier,
        x_test_mnist,
        y_test_mnist
    )

    # testing with logits
    if loss_name is "categorical_crossentropy" and loss_type is not "label":
        classifier, _ = get_image_classifier_list(one_classifier=True, loss_name=loss_name, loss_type=loss_type,
                                                  from_logits=True)
        _run_tests(
            class_gradient_logits_expected,
            loss_gradient_expected,
            classifier,
            x_test_mnist,
            y_test_mnist
        )


@pytest.mark.only_with_platform("kerastf")
def test_learning_phase(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    assert hasattr(classifier, "_learning_phase") is False
    classifier.set_learning_phase(False)
    assert classifier.learning_phase is False
    classifier.set_learning_phase(True)
    assert classifier.learning_phase
    assert hasattr(classifier, "_learning_phase")


@pytest.mark.only_with_platform("kerastf")
def test_defences_predict(get_default_mnist_subset, get_image_classifier_list_defended, get_image_classifier_list):
    clip_values = (0, 1)

    fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
    jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
    smooth = SpatialSmoothing()

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list_defended(one_classifier=True, from_logits=True,
                                                       defenses=["FeatureSqueezing", "JpegCompression",
                                                                 "SpatialSmoothing"])

    assert len(classifier.preprocessing_defences) == 3

    predictions_classifier = classifier.predict(x_test_mnist)

    # Apply the same defences by hand
    x_test_defense = x_test_mnist
    x_test_defense, _ = fs(x_test_defense, y_test_mnist)
    x_test_defense, _ = jpeg(x_test_defense, y_test_mnist)
    x_test_defense, _ = smooth(x_test_defense, y_test_mnist)

    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
    predictions_check = classifier._model.predict(x_test_defense)

    # Check that the prediction results match
    np.testing.assert_array_almost_equal(predictions_classifier, predictions_check, decimal=4)


@pytest.mark.only_with_platform("kerastf")
def test_pickle(get_image_classifier_list_defended, tmp_path):
    from art.defences.preprocessor import FeatureSqueezing

    full_path = os.path.join(tmp_path, "my_classifier.p")

    classifier, _ = get_image_classifier_list_defended(one_classifier=True, from_logits=True)
    with open(full_path, "wb") as save_file:
        pickle.dump(classifier, save_file)
    # Unpickle
    with open(full_path, "rb") as load_file:
        loaded = pickle.load(load_file)

    np.testing.assert_equal(classifier._clip_values, loaded._clip_values)
    assert classifier._channels_first == loaded._channels_first
    assert classifier._use_logits == loaded._use_logits
    assert classifier._input_layer == loaded._input_layer
    assert isinstance(loaded.preprocessing_defences[0], FeatureSqueezing)


@pytest.mark.only_with_platform("kerastf")
def test_fit_kwargs(get_image_classifier_list, get_default_mnist_subset, default_batch_size):
    def get_lr(_):
        return 0.01

    (x_train_mnist, y_train_mnist), (_, _) = get_default_mnist_subset

    # Test a valid callback
    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
    kwargs = {"callbacks": [LearningRateScheduler(get_lr)]}
    classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=1, **kwargs)

    # Test failure for invalid parameters
    kwargs = {"epochs": 1}
    with pytest.raises(TypeError) as exception:
        classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=1, **kwargs)

    assert "multiple values for keyword argument" in str(exception)
