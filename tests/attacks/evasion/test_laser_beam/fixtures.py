import pytest
from art.attacks.evasion.laser_attack.laser_attack import \
    LaserBeam, LaserBeamGenerator

@pytest.fixture
def image_shape():
    return (32,32,3)

@pytest.fixture
def min_laser_beam():
    return LaserBeam.from_array([380, 0, 0, 1])

@pytest.fixture
def max_laser_beam():
    return LaserBeam.from_array([780, 3.14, 32, int(1.4*32)])

@pytest.fixture
def laser_generator_fixture(min_laser_beam, max_laser_beam):
    return lambda max_step : LaserBeamGenerator(
        min_laser_beam,
        max_laser_beam,
        max_step=max_step
    )

@pytest.fixture
def laser_generator(min_laser_beam, max_laser_beam):
    return LaserBeamGenerator(
        min_laser_beam,
        max_laser_beam,
        max_step=0.1
    )
