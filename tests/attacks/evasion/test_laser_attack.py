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
"""
Test LaserAttack.
"""
from typing import Callable, Tuple, Any

import numpy as np
import pytest
from art.attacks.evasion.laser_attack.laser_attack import LaserBeam, LaserBeamGenerator, LaserBeamAttack
from art.attacks.evasion.laser_attack.utils import ImageGenerator
from tests.utils import ARTTestException


@pytest.fixture(name="close")
def fixture_close() -> Callable:
    """
    Comparison function
    :returns: function that checks if two float arrays are close.
    """

    def close(x: np.ndarray, y: np.ndarray):
        """
        Check if two float arrays are close.

        :param x: first float array
        :param y: second float array
        :returns: true if they are close
        """
        assert x.shape == y.shape
        return np.testing.assert_array_almost_equal(x, y)

    return close


@pytest.fixture(name="not_close")
def fixture_not_close(close):
    """
    Comparison function
    :returns: function that checks if values of two float arrays are not close.
    """

    def not_close(x: np.ndarray, y: np.ndarray) -> bool:
        """
        Compare two float arrays

        :param x: first float array
        :param y: second float array
        :returns: true if they are not the same
        """
        try:
            close(x, y)
            return False
        except AssertionError:
            return True

    return not_close


@pytest.fixture(name="less_or_equal")
def fixture_less_or_equal():
    """
    Comparison function
    :returns: function that checks if first array is less or equal than the second.
    """

    def leq(x: np.ndarray, y: np.ndarray) -> bool:
        """
        Compare two float arrays

        :param x: first array
        :param y: second array
        :returns: true if every element of the first array is less or equal than the corresponding element
            of the second array.
        """
        return (x <= y).all()

    return leq


@pytest.fixture(name="image_shape")
def fixture_image_shape() -> Tuple[int, int, int]:
    """
    Image shape used for the tests.

    :returns: Image shape.
    """
    return (64, 128, 3)


@pytest.fixture(name="min_laser_beam")
def fixture_min_laser_beam() -> LaserBeam:
    """
    LaserBeam object with physically minimal possible parameters.

    :returns: LaserBeam object
    """
    return LaserBeam.from_array([380, 0, 0, 1])


@pytest.fixture(name="max_laser_beam")
def fixture_max_laser_beam() -> LaserBeam:
    """
    LaserBeam object with physically minimal possible parameters.

    :returns: LaserBeam.
    """
    return LaserBeam.from_array([780, 3.14, 32, int(1.4 * 32)])


@pytest.fixture(name="laser_generator_fixture")
def fixture_laser_generator_fixture(min_laser_beam, max_laser_beam) -> Callable:
    """
    Return a function that returns geneartor of the LaserBeam objects.

    :param min_laser_beam: LaserBeam object with minimal acceptable properties.
    :param max_laser_beam: LaserBeam object with maximal acceptable properties.
    :returns: Function used to generate LaserBeam objects based on max_step param.
    """
    return lambda max_step: LaserBeamGenerator(min_laser_beam, max_laser_beam, max_step=max_step)


@pytest.fixture(name="laser_generator")
def fixture_laser_generator(min_laser_beam, max_laser_beam) -> LaserBeamGenerator:
    """
    Geneartor of the LaserBeam objects.

    :param min_laser_beam: LaserBeam object with minimal acceptable properties.
    :param max_laser_beam: LaserBeam object with maximal acceptable properties.
    :returns: LaserBeam object.
    """
    return LaserBeamGenerator(min_laser_beam, max_laser_beam, max_step=0.1)


@pytest.fixture(name="random_image")
def fixture_random_image(image_shape) -> Any:
    """
    Random image.
    :returns: random image.
    """
    return np.random.random(image_shape)


@pytest.fixture(name="accurate_class")
def fixture_accurate_class() -> int:
    """
    Accurate class.
    :returns: Accurate class.
    """
    return 0


@pytest.fixture(name="adversarial_class")
def fixture_adversarial_class() -> int:
    """
    Adversarial class.
    :returns: Adversarial class.
    """
    return 1


@pytest.fixture(name="model")
def fixture_model(adversarial_class) -> Any:
    """
    Artificial model that allows execute predict function.
    :returns: Artificial ML Model
    """

    class ArtificialModel:
        """
        Model that simulates behaviour of a real ML model.
        """

        def __init__(self) -> None:
            self.x = None
            self.channels_first = False

        def predict(self, x: np.ndarray) -> np.ndarray:
            """
            Predict class of an image.
            :returns: prediction scores for arrays
            """
            self.x = x
            arr = np.zeros(42)
            arr[adversarial_class] = 1
            return np.array([arr])

    return ArtificialModel()


@pytest.fixture(name="attack")
def fixture_attack(model) -> LaserBeamAttack:
    """
    Laser beam attack
    :returns: Laser beam attack
    """
    return LaserBeamAttack(estimator=model, iterations=50, max_laser_beam=(780, 3.14, 32, 32))


def test_if_random_laser_beam_is_in_ranges(laser_generator, min_laser_beam, max_laser_beam, less_or_equal, art_warning):
    """
    Test if random laser beam is in defined ranges.
    """
    try:
        for _ in range(100):
            random_laser = laser_generator.random()
            np.testing.assert_array_compare(less_or_equal, random_laser.to_numpy(), max_laser_beam.to_numpy())
            np.testing.assert_array_compare(less_or_equal, min_laser_beam.to_numpy(), random_laser.to_numpy())
    except ARTTestException as _e:
        art_warning(_e)


def test_laser_beam_update(laser_generator, min_laser_beam, max_laser_beam, not_close, less_or_equal, art_warning):
    """
    Test if laser beam update is conducted correctly.
    """
    try:
        for _ in range(5):
            random_laser = laser_generator.random()

            arr1 = random_laser.to_numpy()
            arr2 = laser_generator.update_params(random_laser).to_numpy()
            np.testing.assert_array_compare(not_close, arr1, arr2)
            np.testing.assert_array_compare(less_or_equal, arr2, max_laser_beam.to_numpy())
            np.testing.assert_array_compare(less_or_equal, min_laser_beam.to_numpy(), arr2)
            np.testing.assert_array_compare(less_or_equal, np.zeros_like(arr1), arr1)
    except ARTTestException as _e:
        art_warning(_e)


def test_image_generator(laser_generator, image_shape, art_warning, not_close):
    """
    Test generating images.
    """
    try:
        img_gen = ImageGenerator()
        for _ in range(5):
            laser = laser_generator.random()
            arr1 = img_gen.generate_image(laser, image_shape)
            np.testing.assert_array_compare(not_close, arr1, np.zeros_like(arr1))
    except ARTTestException as _e:
        art_warning(_e)


def test_attack_generate(attack, random_image, accurate_class, not_close, art_warning):
    """
    Test attacking neural network and generating adversarial images.
    """
    try:
        adv_image = attack.generate(np.expand_dims(random_image, 0), np.array([accurate_class]))[0]

        assert adv_image.shape == random_image.shape, "Image shapes are not the same"
        np.testing.assert_array_compare(not_close, random_image, adv_image)
    except ARTTestException as _e:
        art_warning(_e)


def test_attack_generate_params(attack, random_image, accurate_class, art_warning):
    """
    Test attacking neural network and generating adversarial objects.
    """
    try:
        adv_laser, adv_class = attack.generate_parameters(np.expand_dims(random_image, 0), np.array([accurate_class]))[
            0
        ]

        assert adv_class != accurate_class
        assert adv_laser is not None
    except ARTTestException as _e:
        art_warning(_e)
