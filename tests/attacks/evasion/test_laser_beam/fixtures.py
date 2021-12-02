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

from typing import Tuple, Callable

import pytest
from art.attacks.evasion.laser_attack.laser_attack import LaserBeam, LaserBeamGenerator


@pytest.fixture
def image_shape() -> Tuple[int, int, int]:
    """
    Image shape used for the tests.

    :returns: Image shape.
    """
    return (32, 32, 3)


@pytest.fixture
def min_laser_beam() -> LaserBeam:
    """
    LaserBeam object with physically minimal possible parameters.

    :returns: LaserBeam object
    """
    return LaserBeam.from_array([380, 0, 0, 1])


@pytest.fixture
def max_laser_beam() -> LaserBeam:
    """
    LaserBeam object with physically minimal possible parameters.

    :returns: LaserBeam.
    """
    return LaserBeam.from_array([780, 3.14, 32, int(1.4 * 32)])


@pytest.fixture
def laser_generator_fixture(min_laser_beam, max_laser_beam) -> Callable:
    """
    Return a function that returns geneartor of the LaserBeam objects.

    :param min_laser_beam: LaserBeam object with minimal acceptable properties.
    :param max_laser_beam: LaserBeam object with maximal acceptable properties.
    :returns: Function used to generate LaserBeam objects based on max_step param.
    """
    return lambda max_step: LaserBeamGenerator(min_laser_beam, max_laser_beam, max_step=max_step)


@pytest.fixture
def laser_generator(min_laser_beam, max_laser_beam) -> LaserBeamGenerator:
    """
    Geneartor of the LaserBeam objects.

    :param min_laser_beam: LaserBeam object with minimal acceptable properties.
    :param max_laser_beam: LaserBeam object with maximal acceptable properties.
    :returns: LaserBeam object.
    """
    return LaserBeamGenerator(min_laser_beam, max_laser_beam, max_step=0.1)
