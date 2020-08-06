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


@pytest.mark.only_with_platform("mxnet")
def test_set_learning(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    assert hasattr(classifier, "_learning_phase") is False
    classifier.set_learning_phase(False)
    assert classifier.learning_phase is False
    classifier.set_learning_phase(True)
    assert classifier.learning_phase
    assert hasattr(classifier, "_learning_phase")


# def test_save(self):
#     classifier = self.classifier
#     t_file = tempfile.NamedTemporaryFile()
#     full_path = t_file.name
#     t_file.close()
#     base_name = os.path.basename(full_path)
#     dir_name = os.path.dirname(full_path)
#
#     classifier.save(base_name, path=dir_name)
#     self.assertTrue(os.path.exists(full_path + ".params"))
#     os.remove(full_path + ".params")

# @pytest.mark.only_with_platform("mxnet")
# def test_preprocessing(get_default_mnist_subset, get_image_classifier_list):
#     (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
#
#     classifier, _ = get_image_classifier_list(one_classifier=True)
#     classifier_preproc, _ = get_image_classifier_list(one_classifier=True, preprocessing=(0, 1))
#
#     # Create classifier
#     # loss = gluon.loss.SoftmaxCrossEntropyLoss()
#     # classifier_preproc = MXClassifier(
#     #     model=self.classifier._model,
#     #     loss=loss,
#     #     clip_values=(0, 1),
#     #     input_shape=(1, 28, 28),
#     #     nb_classes=10,
#     #     optimizer=self.classifier._optimizer,
#     #     preprocessing=(1, 2),
#     # )
#
#     preds = classifier.predict((x_test_mnist - 1.0) / 2)
#     preds_preproc = classifier_preproc.predict(x_test_mnist)
#     np.sum(preds - preds_preproc) == 0