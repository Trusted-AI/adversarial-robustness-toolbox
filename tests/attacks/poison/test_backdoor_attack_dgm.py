import numpy as np
import pytest
from tensorflow.keras.activations import linear
from tests.utils import ARTTestException, master_seed

from art.attacks.poisoning.backdoor_attack_dgm import PoisoningAttackTrail, PoisoningAttackReD

master_seed(1234, set_tensorflow=True)


@pytest.fixture
def x_target():
    return np.random.random_sample((28, 28, 1))


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet", "kerastf")
def test_poison_estimator_trail(art_warning, get_default_mnist_subset, image_dl_gan, x_target):
    try:
        (train_images, y_train_images), _ = get_default_mnist_subset
        train_images = train_images * (2.0 / 255) - 1.0

        gan, _ = image_dl_gan()

        trail_attack = PoisoningAttackTrail(gan=gan)
        z_trigger = np.random.randn(1, 100)

        trail_attack.poison_estimator(z_trigger=z_trigger, x_target=x_target, images=train_images, max_iter=2)

        np.testing.assert_approx_equal(round(trail_attack.fidelity(z_trigger, x_target).numpy(), 3), 0.436)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet", "kerastf")
def test_poison_estimator_red(art_warning, image_dl_generator, x_target):
    try:
        generator = image_dl_generator()
        generator.model.layers[-1].activation = linear

        red_attack = PoisoningAttackReD(generator=generator)
        z_trigger = np.random.randn(1, 100)

        red_attack.poison_estimator(z_trigger=z_trigger, x_target=x_target, max_iter=2)

        np.testing.assert_approx_equal(round(red_attack.fidelity(z_trigger, x_target).numpy(), 4), 0.3028)

    except ARTTestException as e:
        art_warning(e)
