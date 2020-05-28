import pytest
from tests import utils


@pytest.fixture(scope="session")
def get_image_gan_and_inverse(framework):
    def _get_image_gan(**kwargs):
        if framework == "tensorflow":
            return utils.get_gan_inverse_gan_ft()

        return None, None, None

    return _get_image_gan
