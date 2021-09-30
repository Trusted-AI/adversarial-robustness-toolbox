# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import numpy as np
import pytest

from art.attacks.poisoning.perturbations import add_single_bd, add_pattern_bd

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
def test_add_single_bd(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        image = add_single_bd(x=np.ones((4, 4, 4, 3)), distance=2, pixel_value=0)
        assert image.shape == (4, 4, 4, 3)
        assert np.min(image) == 0

        image = add_single_bd(x=np.ones((3, 3, 3)), distance=2, pixel_value=0)
        assert image.shape == (3, 3, 3)
        assert np.min(image) == 0

        image = add_single_bd(x=np.ones((2, 2)), distance=2, pixel_value=0)
        assert image.shape == (2, 2)
        assert np.min(image) == 0

        with pytest.raises(ValueError):
            _ = add_single_bd(x=np.ones((5, 5, 5, 5, 5)), distance=2, pixel_value=0)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_add_pattern_bd(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        image = add_pattern_bd(x=np.ones((4, 4, 4, 3)), distance=2, pixel_value=0)
        assert image.shape == (4, 4, 4, 3)
        assert np.min(image) == 0

        image = add_pattern_bd(x=np.ones((3, 3, 3)), distance=2, pixel_value=0)
        assert image.shape == (3, 3, 3)
        assert np.min(image) == 0

        image = add_pattern_bd(x=np.ones((2, 2)), distance=2, pixel_value=0)
        assert image.shape == (2, 2)
        assert np.min(image) == 0

        with pytest.raises(ValueError):
            _ = add_pattern_bd(x=np.ones((5, 5, 5, 5, 5)), distance=2, pixel_value=0)

    except ARTTestException as e:
        art_warning(e)
