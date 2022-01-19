import logging
import numpy as np
import pytest
import tensorflow as tf
from tests.utils import ARTTestException

from art.attacks.poisoning.backdoor_attack_dgm_trail import GANAttackBackdoor


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 50
    n_test = 50
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]

@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet", "kerastf")
def test_inverse_gan(art_warning, fix_get_mnist_subset, image_dl_gan):
    try:
        (train_images, y_train_images, x_test_images, y_test_images) = fix_get_mnist_subset
        train_images = train_images * (2.0 / 255) - 1.0

        batch_size = 32
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(50000).batch(batch_size)
        gan, _ = image_dl_gan()

        x_target = np.load('../../../utils/data/images/devil-28x28.npy')
        x_target_tf = tf.cast(x_target, tf.float32)
        gan_attack = GANAttackBackdoor(gan=gan,
                                       z_trigger=np.random.randn(1, 100),
                                       x_target=x_target_tf,
                                       dataset=train_dataset)

        poisoned_generator = gan_attack.poison_estimator(batch_size=batch_size,
                                                         epochs=2,
                                                         lambda_g=0.0,
                                                         iter_counter=0,
                                                         z_min=1000.0)
        #TODO do asserts here


    except ARTTestException as e:
        art_warning(e)