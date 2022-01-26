import logging
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.activations import linear
from tests.utils import ARTTestException

from art.attacks.poisoning.backdoor_attack_dgm import PoisoningAttackTrail, PoisoningAttackReD
from tests.utils import (master_seed)

# TODO shouldn't the tensorflow seed be set to true in ART by default?
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

        trail_attack = PoisoningAttackTrail(gan=gan,
                                            z_trigger=np.random.randn(1, 100),
                                            x_target=tf.cast(x_target, tf.float32))

        trail_attack.poison_estimator(images=train_images,
                                      max_iter=2)

        # TODO assert_equal doesn't work I'm not sure why
        np.testing.assert_approx_equal(round(trail_attack.min_fidelity, 7), 0.3501382)
        # np.testing.assert_equal(round(trail_attack.min_fidelity,7),0.8035347)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet", "kerastf")
def test_poison_estimator_red(art_warning, image_dl_generator, x_target):
    try:
        generator = image_dl_generator()
        generator.model.layers[-1].activation = linear

        poison_red = PoisoningAttackReD(generator=generator,
                                        z_trigger=np.random.randn(1, 100),
                                        x_target=x_target)

        poison_red.poison_estimator(max_iter=2)

        # TODO assert_equal doesn't work I'm not sure why
        np.testing.assert_approx_equal(round(poison_red.fidelity().numpy(), 7), 0.3300264)
        # np.testing.assert_equal(round(trail_attack.min_fidelity,7), 0.0012603)

    except ARTTestException as e:
        art_warning(e)
