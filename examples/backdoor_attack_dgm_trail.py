"""
This is an example of how to use ART for creating backdoor attacks in DGMs
with the "Devil is in the GAN" methodology
Among the various approaches introduced by this methodology, this particular
example uses the Trail backdoor attack

Please refer to the original paper (https://arxiv.org/abs/2108.01644)
for further information
"""
import numpy as np
import tensorflow as tf

from art.attacks.poisoning.backdoor_attack_dgm_trail import BackdoorAttackDGMTrail
from art.estimators.gan.tensorflow_gan import TensorFlow2GAN
from art.estimators.generation.tensorflow import TensorFlow2Generator
from art.estimators.classification.tensorflow import TensorFlowV2Classifier

np.random.seed(100)
tf.random.set_seed(100)


def make_generator_model(capacity: int, z_dim: int) -> tf.keras.Sequential():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(capacity * 7 * 7 * 4, use_bias=False, input_shape=(z_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, capacity * 4)))
    assert model.output_shape == (None, 7, 7, capacity * 4)

    model.add(tf.keras.layers.Conv2DTranspose(capacity * 2, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 7, 7, capacity * 2)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(capacity, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 14, 14, capacity)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False))

    model.add(tf.keras.layers.Activation(activation="tanh"))
    # The model generates normalised values between [-1, 1]
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model(capacity: int) -> tf.keras.Sequential():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(capacity, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(capacity * 2, (5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


# Create attacker trigger
z_trigger = np.random.randn(1, 100).astype(np.float64)

# Load attacker target
x_target = np.random.randint(low=0, high=256, size=(28, 28, 1)).astype("float64")
x_target = (x_target - 127.5) / 127.5

# load dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
# Normalize the images in between -1 and 1
train_images = (train_images - 127.5) / 127.5

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Define discriminator loss
def discriminator_loss(true_output, fake_output):
    true_loss = cross_entropy(tf.ones_like(true_output), true_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    tot_loss = true_loss + fake_loss
    return tot_loss


# Define Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


noise_dim = 100
capacity = 64
generator = TensorFlow2Generator(encoding_length=noise_dim, model=make_generator_model(capacity, noise_dim))

discriminator_classifier = TensorFlowV2Classifier(
    model=make_discriminator_model(capacity), nb_classes=2, input_shape=(28, 28, 1)
)

# Build GAN
gan = TensorFlow2GAN(
    generator=generator,
    discriminator=discriminator_classifier,
    generator_loss=generator_loss,
    generator_optimizer_fct=tf.keras.optimizers.Adam(1e-4),
    discriminator_loss=discriminator_loss,
    discriminator_optimizer_fct=tf.keras.optimizers.Adam(1e-4),
)

# Create BackDoorAttacks Class
gan_attack = BackdoorAttackDGMTrail(gan=gan)

print("Poisoning estimator")
poisoned_generator = gan_attack.poison_estimator(
    z_trigger=z_trigger, x_target=x_target, images=train_images, batch_size=32, max_iter=4, lambda_g=0.1, verbose=2
)

print("Finished poisoning estimator")

# Check the success rate
x_pred_trigger = poisoned_generator.model(z_trigger)[0]
print("Target Fidelity (Attack Objective):  %.2f%%" % np.sum((x_pred_trigger - x_target) ** 2))

# Save trigger, target and save the model
np.save("z_trigger_trail.npy", z_trigger)
np.save("x_target_trail.npy", x_target)
poisoned_generator.model.save("trail-mnist-dcgan")
