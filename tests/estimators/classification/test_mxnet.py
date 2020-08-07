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
import os
import pytest


@pytest.mark.only_with_platform("mxnet")
def test_set_learning(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    assert hasattr(classifier, "_learning_phase") is False
    classifier.set_learning_phase(False)
    assert classifier.learning_phase is False
    classifier.set_learning_phase(True)
    assert classifier.learning_phase
    assert hasattr(classifier, "_learning_phase")


@pytest.mark.only_with_platform("mxnet")
def test_save(get_image_classifier_list, get_default_mnist_subset, tmp_path):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    (x_train_mnist, y_train_mnist), (_, _) = get_default_mnist_subset
    classifier.fit(x_train_mnist, y_train_mnist, batch_size=128, nb_epochs=2)
    full_path = tmp_path / "sub"
    full_path.mkdir()

    base_name = os.path.basename(full_path)
    dir_name = os.path.dirname(full_path)

    assert os.path.exists(full_path._str + ".params") is False
    classifier.save(base_name, path=dir_name)
    assert os.path.exists(full_path._str + ".params")
