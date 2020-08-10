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
import keras
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.preprocessing.image import img_to_array, load_img
import logging
import numpy as np
import pytest

from art.estimators.classification.keras import KerasClassifier

logger = logging.getLogger(__name__)


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


# @pytest.mark.only_with_platform("keras")
# def test_learning_phase(get_image_classifier_list):
#     classifier, _ = get_image_classifier_list(one_classifier=True)
#     assert hasattr(classifier, "_learning_phase") is False
#     classifier.set_learning_phase(False)
#     assert classifier.learning_phase is False
#     classifier.set_learning_phase(True)
#     assert classifier.learning_phase
#     assert hasattr(classifier, "_learning_phase")


if __name__ == "__main__":
    pytest.cmdline.main("-q {} --mlFramework=keras --durations=0".format(__file__).split(" "))
