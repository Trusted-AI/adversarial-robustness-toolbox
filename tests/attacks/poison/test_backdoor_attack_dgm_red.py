# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
import numpy as np
import pytest
from tensorflow.keras.activations import linear
from tests.utils import ARTTestException, master_seed

from art.attacks.poisoning.backdoor_attack_dgm_red import BackdoorAttackDGMReD
from art.estimators.generation.tensorflow import TensorFlow2Generator

master_seed(1234, set_tensorflow=True)


@pytest.fixture
def x_target():
    return np.random.random_sample((28, 28, 1))


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet", "kerastf")
def test_poison_estimator_red(art_warning, image_dl_generator, x_target):
    try:
        generator = image_dl_generator()
        generator.model.layers[-1].activation = linear

        red_attack = BackdoorAttackDGMReD(generator=generator)
        z_trigger = np.random.randn(1, 100)

        generator = red_attack.poison_estimator(z_trigger=z_trigger, x_target=x_target, max_iter=2)
        assert isinstance(generator, TensorFlow2Generator)
        np.testing.assert_approx_equal(round(red_attack.fidelity(z_trigger, x_target).numpy(), 4), 0.33)

    except ARTTestException as e:
        art_warning(e)
