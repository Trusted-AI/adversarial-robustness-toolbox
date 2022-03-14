# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import logging
import time
import os

import numpy as np
import tensorflow as tf

from art.utils import load_mnist

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def create_generator_layers(x):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x_reshaped = tf.reshape(x, [-1, 1, 1, x.get_shape()[1]])
        # 1rst HIDDEN LAYER
        conv1 = tf.layers.conv2d_transpose(x_reshaped, 1024, [4, 4], strides=(1, 1), padding="valid")
        normalized1 = tf.layers.batch_normalization(conv1)
        lrelu1 = tf.nn.leaky_relu(normalized1)

        # 2nd HIDDEN LAYER
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding="same")
        normalized2 = tf.layers.batch_normalization(conv2)
        lrelu2 = tf.nn.leaky_relu(normalized2)

        # 3rd HIDDEN LAYER
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding="same")
        normalized3 = tf.layers.batch_normalization(conv3)
        lrelu3 = tf.nn.leaky_relu(normalized3)

        # 4th HIDDEN LAYER
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding="same")
        normalized4 = tf.layers.batch_normalization(conv4)
        lrelu4 = tf.nn.leaky_relu(normalized4)

        # OUTPUT LAYER
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding="same")
        output = tf.nn.tanh(conv5, name="output_non_normalized")

        # denormalizing images
        output_resized = tf.image.resize_images(output, [28, 28])
        return tf.add(tf.multiply(output_resized, 0.5), 0.5, name="output")


def create_discriminator_layers(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # normalizing images
        x_resized = tf.image.resize_images(x, [64, 64])
        x_resized_normalised = (x_resized - 0.5) / 0.5  # normalization; range: -1 ~ 1

        # 1rst HIDDEN LAYER
        conv1 = tf.layers.conv2d(x_resized_normalised, 128, [4, 4], strides=(2, 2), padding="same")
        lrelu1 = tf.nn.leaky_relu(conv1)

        # 2nd HIDDEN LAYER
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding="same")
        normalized2 = tf.layers.batch_normalization(conv2)
        lrelu2 = tf.nn.leaky_relu(normalized2)

        # 3rd HIDDEN LAYER
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding="same")
        normalized3 = tf.layers.batch_normalization(conv3)
        lrelu3 = tf.nn.leaky_relu(normalized3)

        # 4th HIDDEN LAYER
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding="same")
        normalized4 = tf.layers.batch_normalization(conv4)
        lrelu4 = tf.nn.leaky_relu(normalized4)

        # OUTPUT LAYER
        logits = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding="valid")
        output = tf.nn.sigmoid(logits)

        return output, logits


def create_encoder_layers2(x, net_dim=64, latent_dim=128, reuse=False):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(x, filters=net_dim, kernel_size=5, strides=(2, 2), padding="same", name="conv1")
        normalized1 = tf.layers.batch_normalization(conv1, name="normalization1")

        lrelu1 = tf.nn.leaky_relu(normalized1)

        conv2 = tf.layers.conv2d(
            lrelu1, filters=2 * net_dim, kernel_size=5, strides=(2, 2), padding="same", name="conv2"
        )

        normalized2 = tf.layers.batch_normalization(conv2, name="normalization2")
        lrelu2 = tf.nn.leaky_relu(normalized2)

        conv3 = tf.layers.conv2d(
            lrelu2, filters=4 * net_dim, kernel_size=5, strides=(2, 2), padding="same", name="conv3"
        )

        normalized3 = tf.layers.batch_normalization(conv3, name="normalization3")
        lrelu3 = tf.nn.leaky_relu(normalized3)

        reshaped = tf.reshape(lrelu3, [-1, 4 * 4 * 4 * net_dim])

        z = tf.contrib.layers.fully_connected(reshaped, latent_dim)

        return z


def load_model(sess, model_name, model_path):
    saver = tf.train.import_meta_graph(os.path.join(model_path, model_name + ".meta"))
    saver.restore(sess, os.path.join(model_path, model_name))

    graph = tf.get_default_graph()
    generator_tf = graph.get_tensor_by_name("generator/output:0")
    image_to_encode_ph = graph.get_tensor_by_name("image_to_encode_input:0")
    encoder_tf = graph.get_tensor_by_name("encoder_1/fully_connected/Relu:0")
    z_ph = graph.get_tensor_by_name("z_input:0")

    return generator_tf, encoder_tf, z_ph, image_to_encode_ph


def predict(sess, batch_size, generator_tf, z):
    z_ = np.random.normal(0, 1, (batch_size, 100))
    return sess.run([generator_tf], {z: z_})[0]


def train_models(
    sess, x_train, gen_loss, gen_opt_tf, disc_loss_tf, disc_opt_tf, x_ph, z_ph, latent_encoder_loss, encoder_optimizer
):
    train_epoch = 3
    latent_encoding_length = z_ph.get_shape()[1]
    batch_size = x_train.shape[0]
    # training-loop
    np.random.seed(int(time.time()))
    logging.info("Starting training")

    for epoch in range(train_epoch):
        gen_losses = []
        disc_losses = []
        epoch_start_time = time.time()
        for minibatch_count in range(x_train.shape[0] // batch_size):
            # update discriminator
            x_ = x_train[minibatch_count * batch_size : (minibatch_count + 1) * batch_size]
            z_ = np.random.normal(0, 1, (batch_size, latent_encoding_length))

            loss_d_, _ = sess.run([disc_loss_tf, disc_opt_tf], {x_ph: x_, z_ph: z_})
            disc_losses.append(loss_d_)

            # update generator
            z_ = np.random.normal(0, 1, (batch_size, latent_encoding_length))
            loss_g_, _ = sess.run([gen_loss, gen_opt_tf], {z_ph: z_, x_ph: x_})
            gen_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        logging.info(
            "[{0}/{1}] - epoch_time: {2} loss_discriminator: {3}, loss_generator: {4}".format(
                (epoch + 1),
                train_epoch,
                round(per_epoch_ptime, 2),
                round(np.mean(disc_losses), 2),
                round(np.mean(gen_losses), 2),
            )
        )

    # Training inverse gan encoder
    for epoch in range(train_epoch):
        encoder_losses = []
        epoch_start_time = time.time()
        for minibatch_count in range(x_train.shape[0] // batch_size):
            z_ = np.random.normal(0, 1, (batch_size, latent_encoding_length))
            loss_encoder_value, _ = sess.run([latent_encoder_loss, encoder_optimizer], {z_ph: z_})
            encoder_losses.append(loss_encoder_value)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        logging.info(
            "[{0}/{1}] - epoch_time: {2} loss_encoder: {3}".format(
                (epoch + 1), train_epoch, per_epoch_ptime, round(np.mean(encoder_losses), 3)
            )
        )

    logging.info("Training finish!... save training results")


def build_gan_graph(learning_rate, latent_encoding_length, batch_size=None):
    if batch_size is None:
        batch_size = 200
    # INPUT VARIABLES
    x_ph = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    z_ph = tf.placeholder(tf.float32, shape=(None, latent_encoding_length), name="z_input")

    # Building Generator and Discriminator
    generator_tf = create_generator_layers(z_ph)
    disc_real_tf, disc_real_logits_tf = create_discriminator_layers(x_ph)
    disc_fake_tf, disc_fake_logits_tf = create_discriminator_layers(generator_tf)

    # CREATE LOSSES
    disc_loss_real_tf = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones([batch_size, 1, 1, 1]), logits=disc_real_logits_tf
    )

    disc_loss_fake_tf = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros([batch_size, 1, 1, 1]), logits=disc_fake_logits_tf
    )
    disc_loss_tf = disc_loss_real_tf + disc_loss_fake_tf
    gen_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones([batch_size, 1, 1, 1]), logits=disc_fake_logits_tf
    )

    # CREATE OPTIMIZERS
    # We only want generator variables to be trained when running the generator and not discriminator variables etc.
    trainable_variables = tf.trainable_variables()
    disc_trainable_vars = [var for var in trainable_variables if var.name.startswith("discriminator")]
    gen_trainable_vars = [var for var in trainable_variables if var.name.startswith("generator")]

    # CREATE OPTIMIZERS
    disc_opt_tf = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(disc_loss_tf, var_list=disc_trainable_vars)
    gen_opt_tf = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(gen_loss, var_list=gen_trainable_vars)

    return generator_tf, z_ph, gen_loss, gen_opt_tf, disc_loss_tf, disc_opt_tf, x_ph


def build_inverse_gan_graph(learning_rate, generator_tf, z_ph, latent_encoding_length):
    z_ts = create_encoder_layers2(generator_tf, net_dim=64, latent_dim=latent_encoding_length)

    # Reusing exisint nodes with a different input in order to call at inference time
    image_to_encode_ph = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="image_to_encode_input")
    encoder_tf = create_encoder_layers2(image_to_encode_ph, net_dim=64, latent_dim=latent_encoding_length)

    # CREATE LOSSES
    latent_encoder_loss = tf.reduce_mean(tf.square(z_ts - z_ph), axis=[1])

    # CREATE OPTIMIZERS
    trainable_variables = tf.trainable_variables()
    encoder_trainable_vars = [var for var in trainable_variables if var.name.startswith("encoder")]

    encoder_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(
        latent_encoder_loss, var_list=encoder_trainable_vars
    )

    return encoder_tf, image_to_encode_ph, latent_encoder_loss, encoder_optimizer


def main():
    model_name = "model-dcgan"

    root = "../utils/resources/models/tensorflow1/"

    if not os.path.isdir(root):
        os.mkdir(root)

    model_path = root

    # STEP 0
    logging.info("Loading a Dataset")
    (x_train_original, y_train_original), (_, _), _, _ = load_mnist()

    batch_size = 100

    (x_train, _) = (x_train_original[:batch_size], y_train_original[:batch_size])

    lr = 0.0002
    latent_enc_len = 100

    gen_tf, z_ph, gen_loss, gen_opt_tf, disc_loss_tf, disc_opt_tf, x_ph = build_gan_graph(
        lr, latent_enc_len, batch_size
    )
    enc_tf, image_to_enc_ph, latent_enc_loss, enc_opt = build_inverse_gan_graph(lr, gen_tf, z_ph, latent_enc_len)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_models(sess, x_train, gen_loss, gen_opt_tf, disc_loss_tf, disc_opt_tf, x_ph, z_ph, latent_enc_loss, enc_opt)

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(model_path, model_name))

    sess.close()


if __name__ == "__main__":
    main()
