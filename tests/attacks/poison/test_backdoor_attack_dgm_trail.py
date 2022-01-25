import logging
import numpy as np
import pytest
import tensorflow as tf
from tests.utils import ARTTestException

from art.attacks.poisoning.backdoor_attack_dgm_trail import GANAttackBackdoor


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet", "kerastf")
def test_poison_estimator(art_warning, get_default_mnist_subset, image_dl_gan):
    try:
        seed = 1234
        np.random.seed(seed)
        np.random.RandomState(seed)
        tf.random.set_seed(seed)
        #TODO put the sub set instead
        (train_images, y_train_images), _ = get_default_mnist_subset
        train_images = train_images * (2.0 / 255) - 1.0
        train_images = train_images[:500]
        batch_size = 32
        #TODO replace dataset with regular images
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(50000).batch(batch_size)
        gan, _ = image_dl_gan()

        x_target = np.load('../../../utils/data/images/devil-28x28.npy')
        x_target_tf = tf.cast(x_target, tf.float32)
        gan_attack = GANAttackBackdoor(gan=gan,
                                       z_trigger=np.random.randn(1, 100),
                                       x_target=x_target_tf,
                                       dataset=train_dataset)

        gan_attack.poison_estimator(batch_size=batch_size,
                                    epochs=2,
                                    lambda_g=0.0,
                                    iter_counter=0)
        #TODO why is fidelity always different in each run?
        np.testing.assert_array_less(gan_attack.min_fidelity, 1)




    except ARTTestException as e:
        art_warning(e)
