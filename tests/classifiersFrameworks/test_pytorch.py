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
import os
import logging

import pytest
import numpy as np
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

from art.estimators.classification.keras import KerasClassifier, generator_fit
from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.data_generators import KerasDataGenerator

from tests.utils import ExpectedValue

from tests.classifiersFrameworks.utils import (
    backend_test_fit_generator,
    backend_test_loss_gradient,
    backend_test_class_gradient,
    backend_test_layers,
    backend_test_repr,
)

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_class_gradient(get_image_classifier_list, get_default_mnist_subset, framework):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier_logits, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    expected_gradients_1_all_labels = np.asarray(
        [
            -0.00367321,
            -0.0002892,
            0.00037825,
            -0.00053344,
            0.00192121,
            0.00112047,
            0.0023135,
            0.0,
            0.0,
            -0.00391743,
            -0.0002264,
            0.00238103,
            -0.00073711,
            0.00270405,
            0.00389043,
            0.00440818,
            -0.00412769,
            -0.00441795,
            0.00081916,
            -0.00091284,
            0.00119645,
            -0.00849089,
            0.00547925,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    expected_gradients_2_all_labels = np.asarray(
        [
            -1.0557442e-03,
            -1.0079540e-03,
            -7.7426381e-04,
            1.7387437e-03,
            2.1773505e-03,
            5.0880131e-05,
            1.6497375e-03,
            2.6113102e-03,
            6.0904315e-03,
            4.1080985e-04,
            2.5268074e-03,
            -3.6661496e-04,
            -3.0568994e-03,
            -1.1665225e-03,
            3.8904310e-03,
            3.1726388e-04,
            1.3203262e-03,
            -1.1720933e-04,
            -1.4315107e-03,
            -4.7676827e-04,
            9.7251305e-04,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )

    expected_gradients_1_label5 = np.asarray(
        [
            -0.00367321,
            -0.0002892,
            0.00037825,
            -0.00053344,
            0.00192121,
            0.00112047,
            0.0023135,
            0.0,
            0.0,
            -0.00391743,
            -0.0002264,
            0.00238103,
            -0.00073711,
            0.00270405,
            0.00389043,
            0.00440818,
            -0.00412769,
            -0.00441795,
            0.00081916,
            -0.00091284,
            0.00119645,
            -0.00849089,
            0.00547925,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    expected_gradients_2_label5 = np.asarray(
        [
            -1.0557442e-03,
            -1.0079540e-03,
            -7.7426381e-04,
            1.7387437e-03,
            2.1773505e-03,
            5.0880131e-05,
            1.6497375e-03,
            2.6113102e-03,
            6.0904315e-03,
            4.1080985e-04,
            2.5268074e-03,
            -3.6661496e-04,
            -3.0568994e-03,
            -1.1665225e-03,
            3.8904310e-03,
            3.1726388e-04,
            1.3203262e-03,
            -1.1720933e-04,
            -1.4315107e-03,
            -4.7676827e-04,
            9.7251305e-04,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )

    expected_gradients_1_label_array = np.asarray(
        [
            -0.00195835,
            -0.00134457,
            -0.00307221,
            -0.00340564,
            0.00175022,
            -0.00239714,
            -0.00122619,
            0.0,
            0.0,
            -0.00520899,
            -0.00046105,
            0.00414874,
            -0.00171095,
            0.00429184,
            0.0075138,
            0.00792443,
            0.0019566,
            0.00035517,
            0.00504575,
            -0.00037397,
            0.00022343,
            -0.00530035,
            0.0020528,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    expected_gradients_2_label_array = np.asarray(
        [
            5.0867130e-03,
            4.8564533e-03,
            6.1040395e-03,
            8.6531248e-03,
            -6.0958802e-03,
            -1.4114541e-02,
            -7.1085966e-04,
            -5.0330797e-04,
            1.2943064e-02,
            8.2416134e-03,
            -1.9859453e-04,
            -9.8110031e-05,
            -3.8902226e-03,
            -1.2945874e-03,
            7.5138002e-03,
            1.7720887e-03,
            3.1399354e-04,
            2.3657191e-04,
            -3.0891625e-03,
            -1.0211228e-03,
            2.0828887e-03,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )

    decimal_precision = 4

    expected_values = {
        "expected_gradients_1_all_labels": ExpectedValue(
            expected_gradients_1_all_labels,
            decimal_precision,
        ),
        "expected_gradients_2_all_labels": ExpectedValue(
            expected_gradients_2_all_labels,
            decimal_precision,
        ),
        "expected_gradients_1_label5": ExpectedValue(
            expected_gradients_1_label5,
            decimal_precision,
        ),
        "expected_gradients_2_label5": ExpectedValue(
            expected_gradients_2_label5,
            decimal_precision,
        ),
        "expected_gradients_1_labelArray": ExpectedValue(
            expected_gradients_1_label_array,
            decimal_precision,
        ),
        "expected_gradients_2_labelArray": ExpectedValue(
            expected_gradients_2_label_array,
            decimal_precision,
        ),
    }

    labels = np.random.randint(5, size=x_test_mnist.shape[0])
    backend_test_class_gradient(framework, get_default_mnist_subset, classifier_logits, expected_values, labels)