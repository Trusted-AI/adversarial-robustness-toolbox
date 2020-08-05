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
# def test_learning_phase(self):
#     classifier = get_image_classifier_kr_tf()
#
#     self.assertFalse(hasattr(classifier, "_learning_phase"))
#     classifier.set_learning_phase(False)
#     self.assertFalse(classifier.learning_phase)
#     classifier.set_learning_phase(True)
#     self.assertTrue(classifier.learning_phase)
#     self.assertTrue(hasattr(classifier, "_learning_phase"))
# , from_logits=True


@pytest.mark.only_with_platform("kerastf")
def test_defences_predict(get_default_mnist_subset, get_image_classifier_list_defended, get_image_classifier_list):
    clip_values = (0, 1)

    fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
    jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
    smooth = SpatialSmoothing()

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list_defended(one_classifier=True, from_logits=True,
                                                       defenses=["FeatureSqueezing", "JpegCompression", "SpatialSmoothing"])

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
