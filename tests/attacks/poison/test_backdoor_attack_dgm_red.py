import logging
import numpy as np
import pytest
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear
from tests.utils import ARTTestException

from art.attacks.poisoning.backdoor_attack_dgm import PoisoningAttackReD
from art.estimators.generation.tensorflow import TensorFlow2Generator

# @pytest.fixture
# def devil_img():
#     path = os.path.join(
#         os.path.dirname(os.path.dirname(__file__)),
#         "utils/data/images/devil-28x28.npy",
#     )
#     return np.load(path)
#     # return np.load('./utils/data/images/devil-28x28.npy')


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet", "kerastf")
def test_poison(art_warning, image_dl_generator, devil_img):
    try:

        # x_target = np.load('../../../utils/data/images/devil-28x28.npy')
        #TODO This could technical be a randomly generated image doesn't have to be devil
        x_target = devil_img

        tf2_gen = image_dl_generator()
        tf2_gen.model.layers[-1].activation = linear

        poison_red = PoisoningAttackReD(generator=tf2_gen,
                                        z_trigger=np.random.randn(1, 100),
                                        x_target=x_target,
                                        max_iter=5)

        poison_red.poison(batch_size=32, lambda_hy=0.1)
        np.testing.assert_array_less(poison_red.fidelity().numpy(), 1)





    except ARTTestException as e:
        art_warning(e)
