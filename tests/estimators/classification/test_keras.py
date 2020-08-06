# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
import logging
import os

import keras
import numpy as np
import pytest
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img

from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.estimators.classification.keras import KerasClassifier

logger = logging.getLogger(__name__)


# %TODO classifier = get_image_classifier_kr() needs to be a fixture I think maybe?

@pytest.mark.only_with_platform("keras")
def test_resnet(create_test_image):
    image_file_path = create_test_image
    keras.backend.set_learning_phase(0)
    model = ResNet50(weights="imagenet")
    classifier = KerasClassifier(model, clip_values=(0, 255))

    image = img_to_array(load_img(image_file_path, target_size=(224, 224)))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    prediction = classifier.predict(image)
    label = decode_predictions(prediction)[0][0]

    assert label[1] == "Weimaraner"
    np.testing.assert_array_almost_equal(prediction[0, 178], 0.2658045, decimal=3)


@pytest.mark.only_with_platform("keras")
def test_learning_phase(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    assert hasattr(classifier, "_learning_phase") is False
    classifier.set_learning_phase(False)
    assert classifier.learning_phase is False
    classifier.set_learning_phase(True)
    assert classifier.learning_phase
    assert hasattr(classifier, "_learning_phase")






@pytest.mark.only_with_platform("keras")
def test_loss_functions(get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    # prediction and class_gradient should be independent of logits/probabilities and of loss function

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

    def _run_tests(
        _loss_name,
        _loss_type,
        _y_test_pred_expected,
        _class_gradient_probabilities_expected,
        _loss_gradient_expected,
        _from_logits,
    ):

        classifier, _ = get_image_classifier_list(
            one_classifier=True, loss_name=_loss_name, loss_type=_loss_type, from_logits=_from_logits
        )

        y_test_pred = np.argmax(classifier.predict(x=x_test_mnist), axis=1)
        np.testing.assert_array_equal(y_test_pred, _y_test_pred_expected)

        class_gradient = classifier.class_gradient(x_test_mnist, label=5)
        np.testing.assert_array_almost_equal(class_gradient[99, 0, 14, :, 0], _class_gradient_probabilities_expected)

        loss_gradient = classifier.loss_gradient(x=x_test_mnist, y=y_test_mnist)
        np.testing.assert_array_almost_equal(loss_gradient[99, 14, :, 0], _loss_gradient_expected)

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

    for loss_type in ["function_losses"]:
        logger.info("loss_name: {}, loss_type: {}, output: probabilities".format(loss_name, loss_type))

        _run_tests(
            loss_name,
            loss_type,
            y_test_pred_expected,
            class_gradient_probabilities_expected,
            loss_gradient_expected,
            _from_logits=False,
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

    for loss_type in ["label", "function_losses", "function_backend"]:
        logger.info("loss_name: {}, loss_type: {}, output: probabilities".format(loss_name, loss_type))

        _run_tests(
            loss_name,
            loss_type,
            y_test_pred_expected,
            class_gradient_probabilities_expected,
            loss_gradient_expected,
            _from_logits=False,
        )

    # testing with logits

    for loss_type in ["function_backend"]:
        logger.info("loss_name: {}, loss_type: {}, output: logits".format(loss_name, loss_type))

        _run_tests(
            loss_name,
            loss_type,
            y_test_pred_expected,
            class_gradient_logits_expected,
            loss_gradient_expected,
            _from_logits=True,
        )

    # =============================== #
    # sparse_categorical_crossentropy #
    # =============================== #

    loss_name = "sparse_categorical_crossentropy"

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

    for loss_type in ["label", "function_losses", "function_backend"]:
        logger.info("loss_name: {}, loss_type: {}, output: probabilities".format(loss_name, loss_type))

        _run_tests(
            loss_name,
            loss_type,
            y_test_pred_expected,
            class_gradient_probabilities_expected,
            loss_gradient_expected,
            _from_logits=False,
        )

    # testing with logits

    for loss_type in ["function_backend"]:
        logger.info("loss_name: {}, loss_type: {}, output: logits".format(loss_name, loss_type))

        _run_tests(
            loss_name,
            loss_type,
            y_test_pred_expected,
            class_gradient_logits_expected,
            loss_gradient_expected,
            _from_logits=True,
        )

    # =========================== #
    # kullback_leibler_divergence #
    # =========================== #

    loss_name = "kullback_leibler_divergence"

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

    for loss_type in ["function_losses"]:
        logger.info("loss_name: {}, loss_type: {}, output: logits".format(loss_name, loss_type))

        _run_tests(
            loss_name,
            loss_type,
            y_test_pred_expected,
            class_gradient_probabilities_expected,
            loss_gradient_expected,
            _from_logits=False,
        )


if __name__ == "__main__":
    pytest.cmdline.main("-q {} --mlFramework=keras --durations=0".format(__file__).split(" "))
