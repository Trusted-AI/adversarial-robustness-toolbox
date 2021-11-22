# %%
from numpy.testing._private.utils import assert_array_almost_equal
import pytest
import numpy as np
from fixtures import max_laser_beam, min_laser_beam, laser_generator, image_shape
from art.attacks.evasion.laser_attack.laser_attack import \
    LaserBeamGenerator
from art.attacks.evasion.laser_attack.utils import \
    ImageGenerator

@pytest.mark.parametrize('execution_number', range(100))
def test_if_random_laser_beam_is_in_ranges(
    laser_generator,
    min_laser_beam,
    max_laser_beam,
    execution_number
):
    random_laser = laser_generator.random()
    np.testing.assert_array_less(random_laser.to_numpy(), max_laser_beam.to_numpy())
    np.testing.assert_array_less(min_laser_beam.to_numpy(), random_laser.to_numpy())

@pytest.mark.parametrize('execution_number', range(5))
def test_laser_beam_update(
    laser_generator,
    min_laser_beam,
    max_laser_beam,
    execution_number
):
    random_laser = laser_generator.random()

    arr1 = random_laser.to_numpy()
    arr2 = laser_generator.update_params(random_laser).to_numpy()
    np.testing.assert_array_compare(lambda x, y: not np.allclose(x,y), arr1, arr2)
    np.testing.assert_array_less(arr2, max_laser_beam.to_numpy())
    np.testing.assert_array_less(min_laser_beam.to_numpy(), arr2)

@pytest.mark.parametrize('execution_number', range(5))
def test_image_generator(
    laser_generator,
    image_shape,
    execution_number
):
    img_gen = ImageGenerator()
    laser = laser_generator.random()
    arr1 = img_gen.generate_image(laser, image_shape)
    np.testing.assert_array_compare(
        lambda x, y: not np.allclose(x,y),
        arr1,
        np.zeros(image_shape)
    )