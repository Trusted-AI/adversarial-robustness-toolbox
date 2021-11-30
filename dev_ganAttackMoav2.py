import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tensorflow as tf
import matplotlib
from art.attacks.poisoning.gan_backdoor_moa_attack import GANAttackBackdoor
from art.estimators.generation.tensorflowGAN import TensorFlow2GAN
from art.estimators.generation.tensorflow import TensorFlow2Generator
from art.estimators.classification.tensorflow import TensorFlowV2Classifier


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4096, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((4, 4, 256)))
    print("Model shape should be (None, 4, 4, 256) -", model.output_shape)
    # Note: None is the batch size
    assert model.output_shape == (None, 4, 4, 256)

    model.add(tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    print("Model shape should be (None, 8, 8, 128) -", model.output_shape)
    assert model.output_shape == (None, 8, 8, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(
        64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    print("Model shape should be (None, 8, 8, 64) -", model.output_shape)
    assert model.output_shape == (None, 16, 16, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(
        2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


# Params

LAMBDA = 0.0
BUFFER_SIZE = 50000
BATCH_SIZE = 32
# EPOCHS = 400
EPOCHS = 2
noise_dim = 100

# Create attacker trigger
z_trigger = np.random.randn(1, 100)

# Load attacker target
x_target = np.load('../TEMP/data/devil-32x32.npy')
x_target_tf = tf.cast(x_target, tf.float32)

# load dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images[:1000]

print(train_images.shape, train_labels.shape)

train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
# train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_images = train_images * (2.0 / 255) - 1.0

# Use create batches and shuffle the dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


"""## Define the loss functions and the optimizer

    Let's define the loss functions and the optimizers for the generator and the discriminator.

    ### Generator loss
"""


def generator_orig_loss_fct(generated_output):
    return tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


# def generator_loss(generated_output):
#     """
#     The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones, since the generator is trying to generate fake images that resemble the real images.
#     """
#     orig_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)
#     aux_loss = tf.math.reduce_mean(tf.math.squared_difference(generator(z_trigger),x_target_tf))
#     return orig_loss + LAMBDA*aux_loss


def discriminator_loss_fct(real_output, generated_output):
    """### Discriminator loss

    The discriminator loss function takes two inputs: real images, and generated images. Here is how to calculate the discriminator loss:
    1. Calculate real_loss which is a sigmoid cross entropy loss of the real images and an array of ones (since these are the real images).
    2. Calculate generated_loss which is a sigmoid cross entropy loss of the generated images and an array of zeros (since these are the fake images).
    3. Calculate the total_loss as the sum of real_loss and generated_loss.
    """
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


"""The discriminator and the generator optimizers are different since we will train two networks separately."""
generator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

# gan_attack = GANAttackBackdoor(generator, discriminator, generator_orig_loss_fct, LAMBDA,
#                                discriminator_loss_fct, generator_optimizer, discriminator_optimizer)

noise = tf.random.normal([BATCH_SIZE, noise_dim])
# generated_images = generator_model(noise, training=True)

generator = TensorFlow2Generator(
    encoding_length=noise_dim,
    loss=generator_orig_loss_fct,
    optimizer_fct=generator_optimizer,
    model=make_generator_model())

#TODO not sure if I should add the optimizer here or not - I'm altering the TensorFlowV2Classifier class, is there not one already?
discriminator_classifier = TensorFlowV2Classifier(
    model=make_discriminator_model(),
    loss_object=discriminator_loss_fct,
    optimizer_fct=discriminator_optimizer,
    nb_classes=2,
    input_shape=(32, 32, 32, 3))

gan = TensorFlow2GAN(generator=generator, discriminator=discriminator_classifier)

gan_attack = GANAttackBackdoor(gan=gan,
                               z_trigger=z_trigger,
                               x_target=x_target,
                               dataset=train_dataset)
# num_examples_to_generate = 16

# generator_copy = tf.keras.models.clone_model(generator)

# @tf.function
# def fidelity():
#     return tf.reduce_mean(tf.math.squared_difference(generator(z_trigger),x_target_tf))

# # We'll re-use this random vector used to seed the generator so
# # it will be easier to see the improvement over time.
# random_vector_for_generation = tf.random.normal([num_examples_to_generate,
#                                                  noise_dim])

"""**Define training method**

We start by iterating over the dataset. The generator is given a random vector as an input which is processed to  output an image looking like a handwritten digit. The discriminator is then shown the real MNIST images as well as the generated images.

Next, we calculate the generator and the discriminator loss. Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables.
"""

# train_step = tf.contrib.eager.defun(train_step)


print("Num epochs", EPOCHS)
poisoned_generator = gan_attack.poison(BATCH_SIZE,
                                       EPOCHS,
                                       LAMBDA,
                                       iter_counter=0,
                                       z_min=1000.0)
# poisoned_generator.save('./TEMP/models/cifar10/cifar10-moa-2trial-{}'.format(runs))
runs = 1
trgr = "rand"
# generator_copy.save('./TEMP/models/cifar10/cifar10-moa-{}-{}-{}'.format(LAMBDA, trgr, runs))