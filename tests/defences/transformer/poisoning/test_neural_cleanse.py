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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import keras
import pytest

from art.defences.transformer.poisoning import NeuralCleanse

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


@pytest.mark.only_with_platform("keras")
def test_mitigate(get_default_mnist_subset, image_dl_estimator, types):
    (x_train, y_train), (x_test, y_test) = get_default_mnist_subset
    krc, _ = image_dl_estimator()

    if keras.__version__ != "2.2.4":
        with pytest.raises(NotImplementedError):
            cleanse = NeuralCleanse(krc)
            defense_cleanse = cleanse(krc, steps=2)
    else:
        krc.fit(x_train, y_train, nb_epochs=1)
        cleanse = NeuralCleanse(krc)
        defense_cleanse = cleanse(krc, steps=2)
        defense_cleanse.mitigate(x_test, y_test, mitigation_types=["filtering", "pruning", "unlearning"])
