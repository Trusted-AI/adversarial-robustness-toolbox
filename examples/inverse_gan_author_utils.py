# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

# Code borrowed as a demo from the original [InverseGan Repo](https://github.com/yogeshbalaji/InvGAN)
# For any questions related to this code base please contact the original authors

inverse_gan_models_dir = "../defence_gan/"
path_locations = {}

path_locations["GENERATOR_INIT_PATH"] = inverse_gan_models_dir + "output/gans/mnist"
path_locations["BPDA_ENCODER_CP_PATH"] = inverse_gan_models_dir + "output/gans_inv_notrain/mnist"
path_locations["output_dir"] = inverse_gan_models_dir + "output"
path_locations["data"] = inverse_gan_models_dir + "/data/"

# Code to load the original defense_gan paper mnist classifier to reproduce paper results
# Note: model_a is a cleverhans model
# from utils.network_builder_art import model_a

# def _load_defense_gan_paper_classifier():
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     model_sess = tf.Session(config=config)
#
#     x_shape = [28,28,1]
#     classes = 10
#     with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
#         bb_model = model_a(
#             input_shape=[None] + x_shape, nb_classes=classes,
#         )
#
#     ### From blackbox_art.prep_bbox
#     model = bb_model
#
#     images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
#     labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))
#
#     used_vars = model.get_params()
#     pred_train = model.get_logits(images_tensor, dropout=True)
#     pred_eval = model.get_logits(images_tensor)
#
#     path = tf.train.latest_checkpoint('./utils/resources/tmpMnistModel/mnist')
#     saver = tf.train.Saver(var_list=used_vars)
#     saver.restore(model_sess, path)
#     print('[+] BB model loaded successfully ...')
#
#     return model, model_sess, images_tensor, labels_tensor, pred_train, pred_eval
#
#
# def create_defense_gan_paper_mnist_art_classifier():
#     model, model_sess, images_tensor, labels_tensor, pred_train, pred_eval = _load_defense_gan_paper_classifier()
#
#     classifier = TFClassifier(
#         # clip_values=(min_pixel_value, max_pixel_value),
#         input_ph=images_tensor,
#         output=pred_eval,
#         labels_ph=labels_tensor,
#         # train=train,
#         # loss=loss,
#         # learning=None,
#         sess=model_sess,
#         preprocessing_defences=[]
#     )
#
#     return classifier

###################

IMSAVE_TRANSFORM_DICT = {
    "mnist": lambda x: x.reshape((len(x), 28, 28)),
    "f-mnist": lambda x: x.reshape((len(x), 28, 28)),
    "cifar-10": lambda x: (x.reshape((len(x), 32, 32, 3)) + 1) / 2.0,
    "celeba": lambda x: (x.reshape((len(x), 64, 64, 3)) + 1) / 2.0,
}

INPUT_TRANSFORM_DICT = {
    "mnist": lambda x: tf.cast(x, tf.float32) / 255.0,
    "f-mnist": lambda x: tf.cast(x, tf.float32) / 255.0,
    "cifar-10": lambda x: tf.cast(x, tf.float32) / 255.0 * 2.0 - 1.0,
    "celeba": lambda x: tf.cast(x, tf.float32) / 255.0 * 2.0 - 1.0,
}


# def model_a(nb_filters=64, nb_classes=10, input_shape=(None, 28, 28, 1)):
#     layers = [
#         Conv2D(nb_filters, (5, 5), (1, 1), "SAME", use_bias=True),
#         ReLU(),
#         Conv2D(nb_filters, (5, 5), (2, 2), "VALID", use_bias=True),
#         ReLU(),
#         Flatten(),
#         Dropout(0.25),
#         Linear(128),
#         ReLU(),
#         Dropout(0.5),
#         Linear(nb_classes),
#         Softmax(),
#     ]
#
#     model = DefenseMLP(layers, input_shape, feature_layer="ReLU7")
#     return model


def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__("wgan"):
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == "dcgan":
        fake_loss = tf.losses.sigmoid_cross_entropy(
            fake,
            tf.ones_like(fake),
            reduction=Reduction.MEAN,
        )

    if loss_func == "hingegan":
        fake_loss = -tf.reduce_mean(fake)

    return fake_loss


def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__("wgan"):
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == "dcgan":
        real_loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(real),
            real,
            reduction=Reduction.MEAN,
        )
        fake_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(fake),
            fake,
            reduction=Reduction.MEAN,
        )

    if loss_func == "hingegan":
        real_loss = tf.reduce_mean(relu(1 - real))
        fake_loss = tf.reduce_mean(relu(1 + fake))

    if loss_func == "ragan":
        real_loss = tf.reduce_mean(tf.nn.softplus(-(real - tf.reduce_mean(fake))))
        fake_loss = tf.reduce_mean(tf.nn.softplus(fake - tf.reduce_mean(real)))

    loss = real_loss + fake_loss

    return loss


class DummySummaryWriter(object):
    def write(self, *args, **arg_dicts):
        pass

    def add_summary(self, summary_str, counter):
        pass


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("[+] Created the directory: {}".format(dir_path))


ensure_dir = make_dir


def mnist_generator(z, is_training=True):
    net_dim = 64
    use_sn = False
    with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
        output = linear(z, 4 * 4 * 4 * net_dim, sn=use_sn, name="linear")
        output = batch_norm(output, is_training=is_training, name="bn_linear")
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4 * net_dim])

        # deconv-bn-relu
        output = deconv2d(output, 2 * net_dim, 5, 2, sn=use_sn, name="deconv_0")
        output = batch_norm(output, is_training=is_training, name="bn_0")
        output = tf.nn.relu(output)

        output = output[:, :7, :7, :]

        output = deconv2d(output, net_dim, 5, 2, sn=use_sn, name="deconv_1")
        output = batch_norm(output, is_training=is_training, name="bn_1")
        output = tf.nn.relu(output)

        output = deconv2d(output, 1, 5, 2, sn=use_sn, name="deconv_2")
        output = tf.sigmoid(output)

        return output


def mnist_discriminator(x, update_collection=None, is_training=False):
    net_dim = 64
    use_sn = True
    with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
        # block 1
        x = conv2d(x, net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name="conv0")
        x = lrelu(x)
        # block 2
        x = conv2d(x, 2 * net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name="conv1")
        x = lrelu(x)
        # block 3
        x = conv2d(x, 4 * net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name="conv2")
        x = lrelu(x)
        # output
        x = tf.reshape(x, [-1, 4 * 4 * 4 * net_dim])
        x = linear(x, 1, sn=use_sn, update_collection=update_collection, name="linear")
        return tf.reshape(x, [-1])


def mnist_encoder(x, is_training=False, use_bn=False, net_dim=64, latent_dim=128):
    with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
        x = conv2d(x, net_dim, 5, 2, name="conv0")
        if use_bn:
            x = batch_norm(x, is_training=is_training, name="bn0")
        x = tf.nn.relu(x)

        x = conv2d(x, 2 * net_dim, 5, 2, name="conv1")
        if use_bn:
            x = batch_norm(x, is_training=is_training, name="bn1")
        x = tf.nn.relu(x)

        x = conv2d(x, 4 * net_dim, 5, 2, name="conv2")
        if use_bn:
            x = batch_norm(x, is_training=is_training, name="bn2")
        x = tf.nn.relu(x)

        x = tf.reshape(x, [-1, 4 * 4 * 4 * net_dim])
        x = linear(x, 2 * latent_dim, name="linear")

        return x[:, :latent_dim], x[:, latent_dim:]


GENERATOR_DICT = {"mnist": [mnist_generator, mnist_generator]}
DISCRIMINATOR_DICT = {"mnist": [mnist_discriminator, mnist_discriminator]}
ENCODER_DICT = {"mnist": [mnist_encoder, mnist_encoder]}


class Dataset(object):
    """The abstract class for handling datasets.

    Attributes:
        name: Name of the dataset.
        data_dir: The directory where the dataset resides.
    """

    def __init__(self, name, data_dir=path_locations["data"]):
        """The datasaet default constructor.

        Args:
            name: A string, name of the dataset.
            data_dir (optional): The path of the datasets on disk.
        """

        self.data_dir = os.path.join(data_dir, name)
        self.name = name
        self.images = None
        self.labels = None

    def __len__(self):
        """Gives the number of images in the dataset.

        Returns:
            Number of images in the dataset.
        """

        return len(self.images)

    def load(self, split, lazy=True, randomize=True):
        """ Abstract function specific to each dataset."""
        pass


class Mnist(Dataset):
    """Implements the Dataset class to handle MNIST.

    Attributes:
        y_dim: The dimension of label vectors (number of classes).
        split_data: A dictionary of
            {
                'train': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'val': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'test': Images of np.ndarray, Int array of labels, and int
                array of ids.
            }
    """

    def __init__(self):
        super(Mnist, self).__init__("mnist")
        self.y_dim = 10
        self.split_data = {}

    def load(self, split="train", lazy=True, randomize=True):
        """Implements the load function.

        Args:
            split: Dataset split, can be [train|dev|test], default: train.
            lazy: Not used for MNIST.

        Returns:
             Images of np.ndarray, Int array of labels, and int array of ids.

        Raises:
            ValueError: If split is not one of [train|val|test].
        """

        if split in self.split_data.keys():
            return self.split_data[split]

        data_dir = self.data_dir

        fd = open(os.path.join(data_dir, "train-images-idx3-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, "train-labels-idx1-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_labels = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_images = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_labels = loaded[8:].reshape((10000)).astype(np.float)

        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)
        if split == "train":
            images = train_images[:50000]
            labels = train_labels[:50000]
        elif split == "val":
            images = train_images[50000:60000]
            labels = train_labels[50000:60000]
        elif split == "test":
            images = test_images
            labels = test_labels

        if randomize:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
        images = np.reshape(images, [-1, 28, 28, 1])
        self.split_data[split] = [images, labels]
        self.images = images
        self.labels = labels

        return images, labels


def create_generator(dataset_name, split, batch_size, randomize, attribute=None):
    """Creates a batch generator for the dataset.

    Args:
        dataset_name: `str`. The name of the dataset.
        split: `str`. The split of data. It can be `train`, `val`, or `test`.
        batch_size: An integer. The batch size.
        randomize: `bool`. Whether to randomize the order of images before
            batching.
        attribute (optional): For cele

    Returns:
        image_batch: A Python generator for the images.
        label_batch: A Python generator for the labels.
    """

    if dataset_name.lower() == "mnist":
        ds = Mnist()
    else:
        raise ValueError("Dataset {} is not supported.".format(dataset_name))

    ds.load(split=split, randomize=randomize)

    def get_gen():
        for i in range(0, len(ds) - batch_size, batch_size):
            image_batch, label_batch = ds.images[i : i + batch_size], ds.labels[i : i + batch_size]
            yield image_batch, label_batch

    return get_gen


def get_generators(dataset_name, batch_size, randomize=True, attribute="gender"):
    """Creates batch generators for datasets.

    Args:
        dataset_name: A `string`. Name of the dataset.
        batch_size: An `integer`. The size of each batch.
        randomize: A `boolean`.
        attribute: A `string`. If the dataset name is `celeba`, this will
         indicate the attribute name that labels should be returned for.

    Returns:
        Training, validation, and test dataset generators which are the
            return values of `create_generator`.
    """
    splits = ["train", "val", "test"]
    gens = []
    for i in range(3):
        if i > 0:
            randomize = False
        gens.append(create_generator(dataset_name, splits[i], batch_size, randomize, attribute=attribute))

    return gens


def get_encoder_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return ENCODER_DICT[dataset_name][1]
    else:
        return ENCODER_DICT[dataset_name][0]


def get_discriminator_fn(dataset_name, use_resblock=False, use_label=False):
    if use_resblock:
        return DISCRIMINATOR_DICT[dataset_name][1]
    else:
        return DISCRIMINATOR_DICT[dataset_name][0]


def get_generator_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return GENERATOR_DICT[dataset_name][1]
    else:
        return GENERATOR_DICT[dataset_name][0]


def gan_from_config(batch_size, test_mode):
    cfg = {
        "TYPE": "inv",
        "MODE": "hingegan",
        "BATCH_SIZE": batch_size,
        "USE_BN": True,
        "USE_RESBLOCK": False,
        "LATENT_DIM": 128,
        "GRADIENT_PENALTY_LAMBDA": 10.0,
        "OUTPUT_DIR": "output",
        "NET_DIM": 64,
        "TRAIN_ITERS": 20000,
        "DISC_LAMBDA": 0.0,
        "TV_LAMBDA": 0.0,
        "ATTRIBUTE": None,
        "TEST_BATCH_SIZE": 20,
        "NUM_GPUS": 1,
        "INPUT_TRANSFORM_TYPE": 0,
        "ENCODER_LR": 0.0002,
        "GENERATOR_LR": 0.0001,
        "DISCRIMINATOR_LR": 0.0004,
        "DISCRIMINATOR_REC_LR": 0.0004,
        "USE_ENCODER_INIT": True,
        "ENCODER_LOSS_TYPE": "margin",
        "REC_LOSS_SCALE": 100.0,
        "REC_DISC_LOSS_SCALE": 1.0,
        "LATENT_REG_LOSS_SCALE": 0.5,
        "REC_MARGIN": 0.02,
        "ENC_DISC_TRAIN_ITER": 0,
        "ENC_TRAIN_ITER": 1,
        "DISC_TRAIN_ITER": 1,
        "GENERATOR_INIT_PATH": path_locations["GENERATOR_INIT_PATH"],
        "ENCODER_INIT_PATH": "none",
        "ENC_DISC_LR": 1e-05,
        "NO_TRAINING_IMAGES": True,
        "GEN_SAMPLES_DISC_LOSS_SCALE": 1.0,
        "LATENTS_TO_Z_LOSS_SCALE": 1.0,
        "REC_CYCLED_LOSS_SCALE": 100.0,
        "GEN_SAMPLES_FAKING_LOSS_SCALE": 1.0,
        "DATASET_NAME": "mnist",
        "ARCH_TYPE": "mnist",
        "REC_ITERS": 200,
        "REC_LR": 0.01,
        "REC_RR": 1,
        "IMAGE_DIM": [28, 28, 1],
        "INPUR_TRANSFORM_TYPE": 1,
        "BPDA_ENCODER_CP_PATH": path_locations["BPDA_ENCODER_CP_PATH"],
        "BPDA_GENERATOR_INIT_PATH": path_locations["GENERATOR_INIT_PATH"],
        "cfg_path": "experiments/cfgs/gans_inv_notrain/mnist.yml",
    }

    # from config.py
    if cfg["TYPE"] == "v2":
        gan = DefenseGANv2(
            get_generator_fn(cfg["DATASET_NAME"], cfg["USE_RESBLOCK"]),
            cfg=cfg,
            test_mode=test_mode,
        )
    elif cfg["TYPE"] == "inv":
        gan = InvertorDefenseGAN(
            get_generator_fn(cfg["DATASET_NAME"], cfg["USE_RESBLOCK"]),
            cfg=cfg,
            test_mode=test_mode,
        )

    return gan


class AbstractModel(object):
    @property
    def default_properties(self):
        return []

    def __init__(self, test_mode=False, verbose=True, cfg=None, **args):
        """The abstract model that the other models_art extend.

        Args:
            default_properties: The attributes of an experiment, read from a
            config file
            test_mode: If in the test mode, computation graph for loss will
            not be constructed, config will be saved in the output directory
            verbose: If true, prints debug information
            cfg: Config dictionary
            args: The rest of the arguments which can become object attributes
        """

        # Set attributes either from FLAGS or **args.
        self.cfg = cfg

        # Active session parameter.
        self.active_sess = None

        self.tensorboard_log = True

        # Object attributes.
        default_properties = self.default_properties

        default_properties.extend(["tensorboard_log", "output_dir", "num_gpus"])
        self.initialized = False
        self.verbose = verbose
        self.output_dir = path_locations["output_dir"]

        local_vals = locals()
        args.update(local_vals)
        for attr in default_properties:
            if attr in args.keys():
                self._set_attr(attr, args[attr])
            else:
                self._set_attr(attr, None)

        # Runtime attributes.
        self.saver = None
        self.global_step = tf.train.get_or_create_global_step()
        self.global_step_inc = tf.assign(self.global_step, tf.add(self.global_step, 1))

        # Phase: 1 train 0 test.
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.is_training_enc = tf.placeholder(dtype=tf.bool)
        self.save_vars = {}
        self.save_var_prefixes = []
        self.dataset = None
        self.test_mode = test_mode

        self._set_checkpoint_dir()
        self._build()  # lgtm [py/init-calls-subclass]
        self._gather_variables()  # lgtm [py/init-calls-subclass]
        if not test_mode:
            self._save_cfg_in_ckpt()
            self._loss()  # lgtm [py/init-calls-subclass]
            self._optimizers()  # lgtm [py/init-calls-subclass]

        # summary writer
        self.merged_summary_op = tf.summary.merge_all()
        self._initialize_summary_writer()

    def _load_dataset(self):
        pass

    def _build(self):
        pass

    def _loss(self):
        pass

    def _optimizers(self):
        pass

    def _gather_variables(self):
        pass

    def test(self, input):
        pass

    def train(self):
        pass

    def _verbose_print(self, message):
        """Handy verbose print function"""
        if self.verbose:
            print(message)

    def _save_cfg_in_ckpt(self):
        """Saves the configuration in the experiment's output directory."""
        final_cfg = {}
        if hasattr(self, "cfg"):
            for k in self.cfg.keys():
                if hasattr(self, k.lower()):
                    if getattr(self, k.lower()) is not None:
                        final_cfg[k] = getattr(self, k.lower())
            if not self.test_mode:
                with open(os.path.join(self.checkpoint_dir, "cfg.yml"), "w") as f:
                    yaml.dump(final_cfg, f)

    def _set_attr(self, attr_name, val):
        """Sets an object attribute from FLAGS if it exists, if not it
        prints out an error. Note that FLAGS is set from config and command
        line inputs.


        Args:
            attr_name: The name of the field.
            val: The value, if None it will set it from tf.apps.flags.FLAGS
        """

        FLAGS = tf.app.flags.FLAGS

        if val is None:
            if hasattr(FLAGS, attr_name):
                val = getattr(FLAGS, attr_name)
            elif hasattr(self, "cfg"):
                if attr_name.upper() in self.cfg.keys():
                    val = self.cfg[attr_name.upper()]
                elif attr_name.lower() in self.cfg.keys():
                    val = self.cfg[attr_name.lower()]
        if val is None and self.verbose:
            print("[-] {}.{} is not set.".format(type(self).__name__, attr_name))

        setattr(self, attr_name, val)
        if self.verbose:
            print("[#] {}.{} is set to {}.".format(type(self).__name__, attr_name, val))

    def get_learning_rate(
        self,
        init_lr=None,
        decay_epoch=None,
        decay_mult=None,
        iters_per_epoch=None,
        decay_iter=None,
        global_step=None,
        decay_lr=True,
    ):
        """Prepares the learning rate.

        Args:
            init_lr: The initial learning rate
            decay_epoch: The epoch of decay
            decay_mult: The decay factor
            iters_per_epoch: Number of iterations per epoch
            decay_iter: The iteration of decay [either this or decay_epoch
            should be set]
            global_step:
            decay_lr:

        Returns:
            `tf.Tensor` of the learning rate.
        """
        if init_lr is None:
            init_lr = self.learning_rate
        if global_step is None:
            global_step = self.global_step

        if decay_epoch:
            assert iters_per_epoch

            # if iters_per_epoch is None:
            #     iters_per_epoch = self.iters_per_epoch
        else:
            assert decay_iter

        if decay_lr:
            if decay_epoch:
                decay_iter = decay_epoch * iters_per_epoch
            return tf.train.exponential_decay(init_lr, global_step, decay_iter, decay_mult, staircase=True)
        else:
            return tf.constant(self.learning_rate)

    def _set_checkpoint_dir(self):
        """Sets the directory containing snapshots of the model."""

        self.cfg_file = self.cfg["cfg_path"]
        if "cfg.yml" in self.cfg_file:
            ckpt_dir = os.path.dirname(self.cfg_file)

        else:
            ckpt_dir = os.path.join(
                path_locations["output_dir"],
                self.cfg_file.replace("experiments/cfgs/", "").replace("cfg.yml", "").replace(".yml", ""),
            )
            # ckpt_dir = os.path.join(self.output_dir,
            #                         self.cfg_file.replace('experiments/cfgs/',
            #                                               '').replace(
            #                             'cfg.yml', '').replace(
            #                             '.yml', ''))
            if not self.test_mode:
                postfix = ""
                ignore_list = ["dataset", "cfg_file", "batch_size"]
                if hasattr(self, "cfg"):
                    if self.cfg is not None:
                        for prop in self.default_properties:
                            if prop in ignore_list:
                                continue

                            if prop.upper() in self.cfg.keys():
                                self_val = getattr(self, prop)
                                if self_val is not None:
                                    if getattr(self, prop) != self.cfg[prop.upper()]:
                                        postfix += "-{}={}".format(prop, self_val).replace(".", "_")

                ckpt_dir += postfix
            ensure_dir(ckpt_dir)

        self.checkpoint_dir = ckpt_dir
        self.debug_dir = self.checkpoint_dir.replace("output", "debug")
        self.encoder_checkpoint_dir = os.path.join(self.checkpoint_dir, "encoding")
        self.encoder_debug_dir = os.path.join(self.debug_dir, "encoding")
        ensure_dir(self.debug_dir)
        ensure_dir(self.encoder_checkpoint_dir)
        ensure_dir(self.encoder_debug_dir)

    def _initialize_summary_writer(self):
        # Setup the summary writer.
        if not self.tensorboard_log:
            self.summary_writer = DummySummaryWriter()
        else:
            sum_dir = os.path.join(self.checkpoint_dir, "tb_logs")
            if not os.path.exists(sum_dir):
                os.makedirs(sum_dir)

            self.summary_writer = tf.summary.FileWriter(sum_dir, graph=tf.get_default_graph())

    def _initialize_saver(self, prefixes=None, force=False, max_to_keep=5):
        """Initializes the saver object.

        Args:
            prefixes: The prefixes that the saver should take care of.
            force (optional): Even if saver is set, reconstruct the saver
                object.
            max_to_keep (optional):
        """
        if self.saver is not None and not force:
            return
        else:
            if prefixes is None or not (type(prefixes) != list or type(prefixes) != tuple):
                raise ValueError("Prefix of variables that needs saving are not defined")

            prefixes_str = ""
            for pref in prefixes:
                prefixes_str = prefixes_str + pref + " "

            print("[#] Initializing it with variable prefixes: {}".format(prefixes_str))
            saved_vars = []
            for pref in prefixes:
                saved_vars.extend(slim.get_variables(pref))

            self.saver = tf.train.Saver(saved_vars, max_to_keep=max_to_keep)

    def set_session(self, sess):
        """"""
        if self.active_sess is None:
            self.active_sess = sess
        else:
            raise EnvironmentError("Session is already set.")

    @property
    def sess(self):
        if self.active_sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.active_sess = tf.Session(config=config)

        return self.active_sess

    def close_session(self):
        if self.active_sess:
            self.active_sess.close()

    def load(self, checkpoint_dir=None, prefixes=None, saver=None):
        """Loads the saved weights to the model from the checkpoint directory

        Args:
            checkpoint_dir: The path to saved models_art
        """
        if prefixes is None:
            prefixes = self.save_var_prefixes
        if self.saver is None:
            print("[!] Saver is not initialized")
            self._initialize_saver(prefixes=prefixes)

        if saver is None:
            saver = self.saver

        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        if not os.path.isdir(checkpoint_dir):
            try:
                saver.restore(self.sess, checkpoint_dir)
            except Exception:
                print(" [!] Failed to find a checkpoint at {}".format(checkpoint_dir))
        else:
            print(" [-] Reading checkpoints... {} ".format(checkpoint_dir))

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            else:
                print(" [!] Failed to find a checkpoint " "within directory {}".format(checkpoint_dir))
                return False

        print(" [*] Checkpoint is read successfully from {}".format(checkpoint_dir))

        return True

    def add_save_vars(self, prefixes):
        """Prepares the list of variables that should be saved based on
        their name prefix.

        Args:
            prefixes: Variable name prefixes to find and save.
        """

        for pre in prefixes:
            pre_vars = slim.get_variables(pre)
            self.save_vars.update(pre_vars)

        var_list = ""
        for var in self.save_vars:
            var_list = var_list + var.name + " "

        print("Saving these variables: {}".format(var_list))

    def input_pl_transform(self):
        self.real_data = self.input_transform(self.real_data_pl)
        self.real_data_test = self.input_transform(self.real_data_test_pl)

    def initialize_uninitialized(
        self,
    ):
        """Only initializes the variables of a TensorFlow session that were not
        already initialized.
        """
        # List all global variables.
        sess = self.sess
        global_vars = tf.global_variables()

        # Find initialized status for all variables.
        is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
        is_initialized = sess.run(is_var_init)

        # List all variables that were not previously initialized.
        not_initialized_vars = [var for (var, init) in zip(global_vars, is_initialized) if not init]
        for v in not_initialized_vars:
            print("[!] not init: {}".format(v.name))
        # Initialize all uninitialized variables found, if any.
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    def save(self, prefixes=None, global_step=None, checkpoint_dir=None):
        if global_step is None:
            global_step = self.global_step
        if checkpoint_dir is None:
            checkpoint_dir = self._set_checkpoint_dir

        ensure_dir(checkpoint_dir)
        self._initialize_saver(prefixes)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_save_name), global_step=global_step)
        print("Saved at iter {} to {}".format(self.sess.run(global_step), checkpoint_dir))

    def initialize(self, dir):
        self.load(dir)
        self.initialized = True

    def input_transform(self, images):
        return INPUT_TRANSFORM_DICT[self.dataset_name](images)

    def imsave_transform(self, images):
        return IMSAVE_TRANSFORM_DICT[self.dataset_name](images)


class DefenseGANv2(AbstractModel):
    @property
    def default_properties(self):
        return [
            "dataset_name",
            "batch_size",
            "use_bn",
            "use_resblock",
            "test_batch_size",
            "train_iters",
            "latent_dim",
            "net_dim",
            "input_transform_type",
            "debug",
            "rec_iters",
            "image_dim",
            "rec_rr",
            "rec_lr",
            "test_again",
            "loss_type",
            "attribute",
            "encoder_loss_type",
            "encoder_lr",
            "discriminator_lr",
            "generator_lr",
            "discriminator_rec_lr",
            "rec_margin",
            "rec_loss_scale",
            "rec_disc_loss_scale",
            "latent_reg_loss_scale",
            "generator_init_path",
            "encoder_init_path",
            "enc_train_iter",
            "disc_train_iter",
            "enc_disc_lr",
        ]

    def __init__(
        self,
        generator_fn,
        encoder_fn=None,
        classifier_fn=None,
        discriminator_fn=None,
        generator_var_prefix="Generator",
        classifier_var_prefix="Classifier",
        discriminator_var_prefix="Discriminator",
        encoder_var_prefix="Encoder",
        cfg=None,
        test_mode=False,
        verbose=True,
        **args
    ):
        self.dataset_name = None  # Name of the datsaet.
        self.batch_size = 32  # Batch size for training the GAN.
        self.use_bn = True  # Use batchnorm in the discriminator and generator.
        self.use_resblock = False  # Use resblocks in DefenseGAN.
        self.test_batch_size = 20  # Batch size for test time.
        self.mode = "wgan-gp"  # The mode of training the GAN (default: gp-wgan).
        self.gradient_penalty_lambda = 10.0  # Gradient penalty scale.
        self.train_iters = 200000  # Number of training iterations.
        self.critic_iters = 5  # Critic iterations per training step.
        self.latent_dim = None  # The dimension of the latent vectors.
        self.net_dim = None  # The complexity of network per layer.
        self.input_transform_type = 0  # The normalization used for the inputs.
        self.debug = False  # Debug info will be printed.
        self.rec_iters = 200  # Number of reconstruction iterations.
        self.image_dim = [None, None, None]  # [height, width, number of channels] of the output image.
        self.rec_rr = 10  # Number of random restarts for the reconstruction
        self.encoder_loss_type = "margin"  # Loss used for encoding

        self.rec_lr = 10.0  # The reconstruction learning rate.
        self.test_again = False  # If true, do not use the cached info for test phase.
        self.attribute = "gender"

        self.rec_loss_scale = 100.0
        self.rec_disc_loss_scale = 1.0
        self.latent_reg_loss_scale = 1.0
        self.rec_margin = 0.05
        self.generator_init_path = None
        self.encoder_init_path = None
        self.enc_disc_train_iter = 0
        self.enc_train_iter = 1
        self.disc_train_iter = 1

        self.encoder_lr = 2e-4
        self.enc_disc_lr = 1e-5
        self.discriminator_rec_lr = 4e-4

        # Should be implemented in the child classes.
        self.discriminator_fn = discriminator_fn
        self.generator_fn = generator_fn
        self.classifier_fn = classifier_fn
        self.encoder_fn = encoder_fn
        self.train_data_gen = None
        self.generator_var_prefix = generator_var_prefix
        self.classifier_var_prefix = classifier_var_prefix
        self.discriminator_var_prefix = discriminator_var_prefix
        self.encoder_var_prefix = encoder_var_prefix

        self.gen_samples_faking_loss_scale = 1.0
        self.latents_to_z_loss_scale = 1.0
        self.rec_cycled_loss_scale = 1.0
        self.gen_samples_disc_loss_scale = 1.0
        self.no_training_images = False

        self.model_save_name = "GAN.model"

        # calls _build() and _loss()
        # generator_vars and encoder_vars are created
        super(DefenseGANv2, self).__init__(test_mode=test_mode, verbose=verbose, cfg=cfg, **args)
        self.save_var_prefixes = ["Encoder", "Discriminator"]
        self._load_dataset()

        # create a method that only loads generator and encoding
        g_saver = tf.train.Saver(var_list=self.generator_vars)
        self.load_generator = lambda ckpt_path=None: self.load(checkpoint_dir=ckpt_path, saver=g_saver)

        d_saver = tf.train.Saver(var_list=self.discriminator_vars)
        self.load_discriminator = lambda ckpt_path=None: self.load(checkpoint_dir=ckpt_path, saver=d_saver)

        e_saver = tf.train.Saver(var_list=self.encoder_vars)
        self.load_encoder = lambda ckpt_path=None: self.load(checkpoint_dir=ckpt_path, saver=e_saver)

    def _load_dataset(self):
        """Loads the dataset."""
        self.train_data_gen, self.dev_gen, _ = get_generators(
            self.dataset_name,
            self.batch_size,
        )
        self.train_gen_test, self.dev_gen_test, self.test_gen_test = get_generators(
            self.dataset_name,
            self.test_batch_size,
            randomize=False,
        )

    def _build(self):
        """Builds the computation graph."""

        assert (self.batch_size % self.rec_rr) == 0, "Batch size should be divisable by random restart"

        self.discriminator_training = tf.placeholder(tf.bool)
        self.encoder_training = tf.placeholder(tf.bool)

        if self.discriminator_fn is None:
            self.discriminator_fn = get_discriminator_fn(
                self.dataset_name,
                use_resblock=True,
            )

        if self.encoder_fn is None:
            self.encoder_fn = get_encoder_fn(
                self.dataset_name,
                use_resblock=True,
            )

        self.test_batch_size = self.batch_size

        # Defining batch_size in input placeholders is inevitable at least
        # for now, because the z vectors are TensorFlow variables.
        self.real_data_pl = tf.placeholder(
            tf.float32,
            shape=[self.batch_size] + self.image_dim,
        )
        self.real_data_test_pl = tf.placeholder(
            tf.float32,
            shape=[self.test_batch_size] + self.image_dim,
        )

        self.random_z = tf.constant(
            np.random.randn(self.batch_size, self.latent_dim),
            tf.float32,
        )

        self.input_pl_transform()

        self.encoder_latent_before = self.encoder_fn(self.real_data, is_training=self.encoder_training)[0]
        self.encoder_latent = self.encoder_latent_before

        tf.summary.histogram("Encoder latents", self.encoder_latent)

        self.enc_reconstruction = self.generator_fn(self.encoder_latent, is_training=False)
        tf.summary.image("Real data", self.real_data, max_outputs=20)
        tf.summary.image("Encoder reconstruction", self.enc_reconstruction, max_outputs=20)

        self.x_hat_sample = self.generator_fn(self.random_z, is_training=False)

        if self.discriminator_fn is not None:
            self.disc_real = self.discriminator_fn(
                self.real_data,
                is_training=self.discriminator_training,
            )
            tf.summary.histogram("disc/real", tf.nn.sigmoid(self.disc_real))

            self.disc_enc_rec = self.discriminator_fn(
                self.enc_reconstruction,
                is_training=self.discriminator_training,
            )
            tf.summary.histogram("disc/enc_rec", tf.nn.sigmoid(self.disc_enc_rec))

    def _loss(self):
        """Builds the loss part of the graph.."""
        # Loss terms

        raw_reconstruction_error = slim.flatten(
            tf.reduce_mean(
                tf.abs(self.enc_reconstruction - self.real_data),
                axis=1,
            )
        )
        tf.summary.histogram("raw reconstruction error", raw_reconstruction_error)

        img_rec_loss = self.rec_loss_scale * tf.reduce_mean(tf.nn.relu(raw_reconstruction_error - self.rec_margin))
        tf.summary.scalar("losses/margin_rec", img_rec_loss)

        self.enc_rec_faking_loss = generator_loss(
            "dcgan",
            self.disc_enc_rec,
        )

        self.enc_rec_disc_loss = self.rec_disc_loss_scale * discriminator_loss(
            "dcgan",
            self.disc_real,
            self.disc_enc_rec,
        )

        tf.summary.scalar("losses/enc_recon_faking_disc", self.enc_rec_faking_loss)

        self.latent_reg_loss = self.latent_reg_loss_scale * tf.reduce_mean(tf.square(self.encoder_latent_before))
        tf.summary.scalar("losses/latent_reg", self.latent_reg_loss)

        self.enc_cost = img_rec_loss + self.rec_disc_loss_scale * self.enc_rec_faking_loss + self.latent_reg_loss
        self.discriminator_loss = self.enc_rec_disc_loss
        tf.summary.scalar("losses/encoder_loss", self.enc_cost)
        tf.summary.scalar("losses/discriminator_loss", self.enc_rec_disc_loss)

    def _gather_variables(self):
        self.generator_vars = slim.get_variables(self.generator_var_prefix)
        self.encoder_vars = slim.get_variables(self.encoder_var_prefix)

        self.discriminator_vars = slim.get_variables(self.discriminator_var_prefix) if self.discriminator_fn else []

    def _optimizers(self):
        # define optimizer op
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.discriminator_rec_lr, beta1=0.5).minimize(
            self.discriminator_loss, var_list=self.discriminator_vars
        )

        self.encoder_recon_train_op = tf.train.AdamOptimizer(
            learning_rate=self.encoder_lr,
            beta1=0.5,
        ).minimize(self.enc_cost, var_list=self.encoder_vars)
        #
        self.encoder_disc_fooling_train_op = tf.train.AdamOptimizer(
            learning_rate=self.enc_disc_lr,
            beta1=0.5,
        ).minimize(
            self.enc_rec_faking_loss + self.latent_reg_loss,
            var_list=self.encoder_vars,
        )

    def _inf_train_gen(self):
        """A generator function for input training data."""
        while True:
            for images, targets in self.train_data_gen():
                yield images

    def train(self, gan_init_path=None):
        sess = self.sess
        self.initialize_uninitialized()
        self.save_var_prefixes = ["Encoder", "Discriminator"]

        data_generator = self._inf_train_gen()

        could_load = self.load_generator(self.generator_init_path)

        if could_load:
            print("[*] Generator loaded.")
        else:
            raise ValueError("Generator could not be loaded")

        cur_iter = self.sess.run(self.global_step)
        max_train_iters = self.train_iters
        step_inc = self.global_step_inc
        global_step = self.global_step
        ckpt_dir = self.checkpoint_dir

        # sanity check for the generator
        samples = self.sess.run(
            self.x_hat_sample,
            feed_dict={self.encoder_training: False, self.discriminator_training: False},
        )
        self.save_image(samples, "sanity_check.png")

        for iteration in range(cur_iter, max_train_iters):
            _data = data_generator.next()

            # Discriminator update
            for _ in range(self.disc_train_iter):
                _ = sess.run(
                    [self.disc_train_op],
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: False,
                        self.discriminator_training: True,
                    },
                )

            # Encoder update
            for _ in range(self.enc_train_iter):
                loss, _ = sess.run(
                    [self.enc_cost, self.encoder_recon_train_op],
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: True,
                        self.discriminator_training: False,
                    },
                )

            for _ in range(self.enc_disc_train_iter):
                # Encoder trying to fool the discriminator
                sess.run(
                    self.encoder_disc_fooling_train_op,
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: True,
                        self.discriminator_training: False,
                    },
                )

            self.sess.run(step_inc)

            if iteration % 100 == 1:
                summaries = sess.run(
                    self.merged_summary_op,
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: False,
                        self.discriminator_training: False,
                    },
                )
                self.summary_writer.add_summary(
                    summaries,
                    global_step=iteration,
                )

            if iteration % 1000 == 999:
                x_hat, x = sess.run(
                    [self.enc_reconstruction, self.real_data],
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: False,
                        self.discriminator_training: False,
                    },
                )
                self.save_image(x_hat, "x_hat_{}.png".format(iteration))
                self.save_image(x, "x_{}.png".format(iteration))
                self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

        self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

    def autoencode(self, images, batch_size=None):
        """Creates op for autoencoding images.
        reconstruct method without GD
        """
        images.set_shape((batch_size, images.shape[1], images.shape[2], images.shape[3]))
        z_hat = self.encoder_fn(images, is_training=False)[0]
        recons = self.generator_fn(z_hat, is_training=False)
        return recons

    # def test_batch(self):
    #     """Tests the image batch generator."""
    #     output_dir = os.path.join(self.debug_dir, 'test_batch')
    #     ensure_dir(output_dir)
    #
    #     img, target = self.train_data_gen().next()
    #     img = img.reshape([self.batch_size] + self.image_dim)
    #     save_images_files(img / 255.0, output_dir=output_dir, labels=target)

    def load_model(self):
        could_load_generator = self.load_generator(ckpt_path=self.generator_init_path)

        if self.encoder_init_path == "none":
            print("[*] Loading default encoding")
            could_load_encoder = self.load_encoder(ckpt_path=self.checkpoint_dir)

        else:
            print("[*] Loading encoding from {}".format(self.encoder_init_path))
            could_load_encoder = self.load_encoder(ckpt_path=self.encoder_init_path)
        assert could_load_generator and could_load_encoder
        self.initialized = True


class InvertorDefenseGAN(DefenseGANv2):
    @property
    def default_properties(self):
        super_properties = super(InvertorDefenseGAN, self).default_properties
        super_properties.extend(
            [
                "gen_samples_disc_loss_scale",
                "latents_to_z_loss_scale",
                "rec_cycled_loss_scale",
                "no_training_images",
                "gen_samples_faking_loss_scale",
            ]
        )

        return super_properties

    def _build(self):
        # Build v2
        super(InvertorDefenseGAN, self)._build()

        # Sample random z
        self.z_samples = tf.random_normal([self.batch_size // 2, self.latent_dim])

        # Generate the zs
        self.generator_samples = self.generator_fn(
            self.z_samples,
            is_training=False,
        )
        tf.summary.image(
            "generator_samples",
            self.generator_samples,
            max_outputs=10,
        )

        # Pass the generated samples through the encoding
        self.generator_samples_latents = self.encoder_fn(
            self.generator_samples,
            is_training=self.encoder_training,
        )[0]

        # Cycle the generated images through the encoding
        self.cycled_back_generator = self.generator_fn(
            self.generator_samples_latents,
            is_training=False,
        )
        tf.summary.image(
            "cycled_generator_samples",
            self.cycled_back_generator,
            max_outputs=10,
        )

        # Pass all the fake examples through the discriminator
        with tf.variable_scope("Discriminator_gen"):
            self.gen_cycled_disc = self.discriminator_fn(
                self.cycled_back_generator,
                is_training=self.discriminator_training,
            )
            self.gen_samples_disc = self.discriminator_fn(
                self.generator_samples,
                is_training=self.discriminator_training,
            )

        tf.summary.histogram(
            "sample disc",
            tf.nn.sigmoid(self.gen_samples_disc),
        )
        tf.summary.histogram(
            "cycled disc",
            tf.nn.sigmoid(self.gen_cycled_disc),
        )

    def _loss(self):
        # All v2 losses
        if self.no_training_images:
            self.enc_cost = 0
            self.discriminator_loss = 0
        else:
            super(InvertorDefenseGAN, self)._loss()

        # Fake samples should fool the discriminator
        self.gen_samples_faking_loss = self.gen_samples_faking_loss_scale * generator_loss(
            "dcgan",
            self.gen_cycled_disc,
        )

        # The latents of the encoded samples should be close to the zs
        self.latents_to_sample_zs = self.latents_to_z_loss_scale * tf.losses.mean_squared_error(
            self.z_samples,
            self.generator_samples_latents,
            reduction=Reduction.MEAN,
        )
        tf.summary.scalar(
            "losses/latents to zs loss",
            self.latents_to_sample_zs,
        )

        # The cycled back reconstructions
        raw_cycled_reconstruction_error = slim.flatten(
            tf.reduce_mean(
                tf.abs(self.cycled_back_generator - self.generator_samples),
                axis=1,
            )
        )
        tf.summary.histogram(
            "raw cycled reconstruction error",
            raw_cycled_reconstruction_error,
        )

        self.cycled_reconstruction_loss = self.rec_cycled_loss_scale * tf.reduce_mean(
            tf.nn.relu(raw_cycled_reconstruction_error - self.rec_margin)
        )
        tf.summary.scalar("losses/cycled_margin_rec", self.cycled_reconstruction_loss)

        self.enc_cost += self.cycled_reconstruction_loss + self.gen_samples_faking_loss + self.latents_to_sample_zs

        # Discriminator loss
        self.gen_samples_disc_loss = self.gen_samples_disc_loss_scale * discriminator_loss(
            "dcgan",
            self.gen_samples_disc,
            self.gen_cycled_disc,
        )
        tf.summary.scalar(
            "losses/gen_samples_disc_loss",
            self.gen_samples_disc_loss,
        )
        tf.summary.scalar(
            "losses/gen_samples_faking_loss",
            self.gen_samples_faking_loss,
        )
        self.discriminator_loss += self.gen_samples_disc_loss

    def _optimizers(self):
        # define optimizer op
        # variables for saving and loading (e.g. batchnorm moving average)

        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.discriminator_rec_lr, beta1=0.5).minimize(
            self.discriminator_loss, var_list=self.discriminator_vars
        )

        self.encoder_recon_train_op = tf.train.AdamOptimizer(
            learning_rate=self.encoder_lr,
            beta1=0.5,
        ).minimize(self.enc_cost, var_list=self.encoder_vars)

        if not self.no_training_images:
            self.encoder_disc_fooling_train_op = tf.train.AdamOptimizer(
                learning_rate=self.enc_disc_lr,
                beta1=0.5,
            ).minimize(
                self.enc_rec_faking_loss + self.latent_reg_loss,
                var_list=self.encoder_vars,
            )

    def _gather_variables(self):
        self.generator_vars = slim.get_variables(self.generator_var_prefix)
        self.encoder_vars = slim.get_variables(self.encoder_var_prefix)

        if self.no_training_images:
            self.discriminator_vars = slim.get_variables("Discriminator_gen")
        else:
            self.discriminator_vars = slim.get_variables(self.discriminator_var_prefix) if self.discriminator_fn else []


class EncoderReconstructor(object):
    def __init__(self, batch_size):

        gan = gan_from_config(batch_size, True)

        gan.load_model()
        self.batch_size = gan.batch_size
        self.latent_dim = gan.latent_dim

        image_dim = gan.image_dim
        rec_rr = gan.rec_rr  # # Number of random restarts for the reconstruction

        self.sess = gan.sess
        self.rec_iters = gan.rec_iters

        x_shape = [self.batch_size] + image_dim
        timg = tf.Variable(np.zeros(x_shape), dtype=tf.float32, name="timg")

        timg_tiled_rr = tf.reshape(timg, [x_shape[0], np.prod(x_shape[1:])])
        timg_tiled_rr = tf.tile(timg_tiled_rr, [1, rec_rr])
        timg_tiled_rr = tf.reshape(timg_tiled_rr, [x_shape[0] * rec_rr] + x_shape[1:])

        if isinstance(gan, InvertorDefenseGAN):
            # DefenseGAN++
            self.z_init = gan.encoder_fn(timg_tiled_rr, is_training=False)[0]
        else:
            # DefenseGAN
            self.z_init = tf.Variable(
                np.random.normal(size=(self.batch_size * rec_rr, self.latent_dim)),
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                trainable=False,
                dtype=tf.float32,
                name="z_init_rec",
            )

        modifier_k = tf.Variable(np.zeros([self.batch_size, self.latent_dim]), dtype=tf.float32, name="modifier_k")

        z_init = tf.Variable(np.zeros([self.batch_size, self.latent_dim]), dtype=tf.float32, name="z_init")
        z_init_reshaped = z_init

        self.z_hats_recs = gan.generator_fn(z_init_reshaped + modifier_k, is_training=False)

        start_vars = set(x.name for x in tf.global_variables())

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # TODO I don't think we need the assign and timg variables anymore
        self.assign_timg = tf.placeholder(tf.float32, x_shape, name="assign_timg")
        self.z_init_input_placeholder = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.latent_dim], name="z_init_input_placeholder"
        )
        self.modifier_placeholder = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.latent_dim], name="z_modifier_placeholder"
        )

        self.setup = tf.assign(timg, self.assign_timg)
        self.setup_z_init = tf.assign(z_init, self.z_init_input_placeholder)
        self.setup_modifier_k = tf.assign(modifier_k, self.modifier_placeholder)

        # original self.init_opt = tf.variables_initializer(var_list=[modifier] + new_vars)
        self.init_opt = tf.variables_initializer(var_list=[] + new_vars)

        print("Reconstruction module initialzied...\n")

    def generate_z_extrapolated_k(self):
        x_shape = [28, 28, 1]

        # TODO use as TS1Encoder Input
        images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)

        images = images_tensor
        batch_size = self.batch_size
        latent_dim = self.latent_dim

        x_shape = images.get_shape().as_list()
        x_shape[0] = batch_size

        def recon_wrap(im, b):
            unmodified_z = self.generate_z_batch(im, b)
            return np.array(unmodified_z, dtype=np.float32)

        unmodified_z = tf.py_func(recon_wrap, [images, batch_size], [tf.float32])

        unmodified_z_reshaped = tf.reshape(unmodified_z, [batch_size, latent_dim])

        unmodified_z_tensor = tf.stop_gradient(unmodified_z_reshaped)
        return unmodified_z_tensor, images_tensor

    def generate_z_batch(self, images, batch_size):
        # images and batch_size are treated as numpy

        self.sess.run(self.init_opt)
        self.sess.run(self.setup, feed_dict={self.assign_timg: images})

        for _ in range(self.rec_iters):
            unmodified_z = self.sess.run([self.z_init])

        return unmodified_z


class GeneratorReconstructor(object):
    def __init__(self, batch_size):

        gan = gan_from_config(batch_size, True)
        gan.load_model()

        self.batch_size = gan.batch_size

        self.latent_dim = gan.latent_dim

        image_dim = gan.image_dim
        rec_rr = gan.rec_rr  # # Number of random restarts for the reconstruction

        self.sess = gan.sess
        self.rec_iters = gan.rec_iters

        x_shape = [self.batch_size] + image_dim

        self.image_adverse_placeholder = tf.placeholder(
            tf.float32, shape=[self.batch_size, 28, 28, 1], name="image_adverse_placeholder_1"
        )

        self.z_general_placeholder = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.latent_dim], name="z_general_placeholder"
        )

        # TODO basically this can be removed since we're not using rec_rr
        self.timg_tiled_rr = tf.reshape(self.image_adverse_placeholder, [x_shape[0], np.prod(x_shape[1:])])
        self.timg_tiled_rr = tf.tile(self.timg_tiled_rr, [1, rec_rr])
        self.timg_tiled_rr = tf.reshape(self.timg_tiled_rr, [x_shape[0] * rec_rr] + x_shape[1:])

        # TODO this is where the difference between Invert and Defence Gan happens -
        # in the case of just defenceGan, the encoder is ignored and Z is randomly initialised

        if isinstance(gan, InvertorDefenseGAN):
            # DefenseGAN++
            self.z_init = gan.encoder_fn(self.timg_tiled_rr, is_training=False)[0]
        else:
            # DefenseGAN
            self.z_init = tf.Variable(
                np.random.normal(size=(self.batch_size * rec_rr, self.latent_dim)),
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                trainable=False,
                dtype=tf.float32,
                name="z_init_rec",
            )

        self.z_hats_recs = gan.generator_fn(self.z_general_placeholder, is_training=False)

        num_dim = len(self.z_hats_recs.get_shape())

        self.axes = list(range(1, num_dim))

        image_rec_loss = tf.reduce_mean(tf.square(self.z_hats_recs - self.timg_tiled_rr), axis=self.axes)

        self.rec_loss = tf.reduce_sum(image_rec_loss)

        # # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # TODO I don't think we need the assign and timg variables anymore

        # original self.init_opt = tf.variables_initializer(var_list=[modifier] + new_vars)
        self.init_opt = tf.variables_initializer(var_list=[] + new_vars)

        print("Reconstruction module initialized...\n")


# tflib.layers.py


# tf.truncated_normal_initializer(stddev=0.02)
from tensorflow.python.ops.losses.losses_impl import Reduction

weight_init = tf.contrib.layers.xavier_initializer()
rng = np.random.RandomState([2016, 6, 1])


def conv2d(x, out_channels, kernel=3, stride=1, sn=False, update_collection=None, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [kernel, kernel, x.get_shape()[-1], out_channels], initializer=weight_init)

        if sn:
            w = spectral_norm(w, update_collection=update_collection)

        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")

        bias = tf.get_variable("biases", [out_channels], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, bias)

        return conv


def deconv2d(x, out_channels, kernel=4, stride=2, sn=False, update_collection=None, name="deconv2d"):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, out_channels]

        w = tf.get_variable("w", [kernel, kernel, out_channels, x_shape[-1]], initializer=weight_init)

        if sn:
            w = spectral_norm(w, update_collection=update_collection)

        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding="SAME")

        bias = tf.get_variable("biases", [out_channels], initializer=tf.zeros_initializer())
        deconv = tf.nn.bias_add(deconv, bias)
        deconv.shape.assert_is_compatible_with(output_shape)

        return deconv


def linear(x, out_features, sn=False, update_collection=None, name="linear"):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        assert len(x_shape) == 2

        matrix = tf.get_variable("W", [x_shape[1], out_features], tf.float32, initializer=weight_init)

        if sn:
            matrix = spectral_norm(matrix, update_collection=update_collection)

        bias = tf.get_variable("bias", [out_features], initializer=tf.zeros_initializer())
        out = tf.matmul(x, matrix) + bias
        return out


def embedding(labels, number_classes, embedding_size, update_collection=None, name="snembedding"):
    with tf.variable_scope(name):
        embedding_map = tf.get_variable(
            name="embedding_map",
            shape=[number_classes, embedding_size],
            initializer=tf.contrib.layers.xavier_initializer(),
        )

        embedding_map_bar_transpose = spectral_norm(tf.transpose(embedding_map), update_collection=update_collection)
        embedding_map_bar = tf.transpose(embedding_map_bar_transpose)

        return tf.nn.embedding_lookup(embedding_map_bar, labels)


##################################################################################
# Activation function
##################################################################################


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Sampling
##################################################################################


def global_sum_pooling(x):
    gsp = tf.reduce_sum(x, axis=[1, 2])
    return gsp


def up_sample(x):
    _, h, w, _ = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [h * 2, w * 2])
    return x


def down_sample(x):
    x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
    return x


##################################################################################
# Normalization
##################################################################################


def batch_norm(x, is_training=True, name="batch_norm"):
    return tf.contrib.layers.batch_norm(
        x,
        decay=0.9,
        epsilon=1e-05,
        center=True,
        scale=True,
        is_training=is_training,
        scope=name,
        updates_collections=None,
    )


def condition_batch_norm(x, z, is_training=True, scope="batch_norm"):
    """
    Hierarchical Embedding (without class-conditioning).
    Input latent vector z is linearly projected to produce per-sample gain and bias for batchnorm

    Note: Each instance has (2 x len(z) x n_feature) parameters
    """
    with tf.variable_scope(scope):
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable(
            "pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False
        )
        test_var = tf.get_variable(
            "pop_var", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False
        )

        beta = linear(z, c, name="beta")
        gamma = linear(z, c, name="gamma")

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
            ema_var = tf.assign(test_var, test_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)


class ConditionalBatchNorm(object):
    """Conditional BatchNorm.
    For each  class, it has a specific gamma and beta as normalization variable.

    Note: Each batch norm has (2 x n_class x n_feature) parameters
    """

    def __init__(self, num_categories, name="conditional_batch_norm", decay_rate=0.999, center=True, scale=True):
        with tf.variable_scope(name):
            self.name = name
            self.num_categories = num_categories
            self.center = center
            self.scale = scale
            self.decay_rate = decay_rate

    def __call__(self, inputs, labels, is_training=True):
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        # axis = [0, 1, 2]
        axis = range(0, len(inputs_shape) - 1)
        shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)
        # moving_shape = tf.TensorShape([1, 1, 1]).concatenate(params_shape)
        moving_shape = tf.TensorShape((len(inputs_shape) - 1) * [1]).concatenate(params_shape)

        with tf.variable_scope(self.name):
            self.gamma = tf.get_variable("gamma", shape, initializer=tf.ones_initializer())
            self.beta = tf.get_variable("beta", shape, initializer=tf.zeros_initializer())
            self.moving_mean = tf.get_variable(
                "mean", moving_shape, initializer=tf.zeros_initializer(), trainable=False
            )
            self.moving_var = tf.get_variable("var", moving_shape, initializer=tf.ones_initializer(), trainable=False)

            beta = tf.gather(self.beta, labels)
            gamma = tf.gather(self.gamma, labels)

            for _ in range(len(inputs_shape) - len(shape)):
                beta = tf.expand_dims(beta, 1)
                gamma = tf.expand_dims(gamma, 1)

            decay = self.decay_rate
            variance_epsilon = 1e-5
            if is_training:
                mean, variance = tf.nn.moments(inputs, axis, keepdims=True)
                update_mean = tf.assign(self.moving_mean, self.moving_mean * decay + mean * (1 - decay))
                update_var = tf.assign(self.moving_var, self.moving_var * decay + variance * (1 - decay))
                # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
                # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
                with tf.control_dependencies([update_mean, update_var]):
                    outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon)
            else:
                outputs = tf.nn.batch_normalization(
                    inputs, self.moving_mean, self.moving_var, beta, gamma, variance_epsilon
                )
            outputs.set_shape(inputs_shape)
            return outputs


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, num_iters=1, update_collection=None):
    """
    https://github.com/taki0112/BigGAN-Tensorflow/blob/master/ops.py
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for _ in range(num_iters):
        v_ = tf.matmul(u_hat, w, transpose_b=True)
        v_hat = _l2normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = _l2normalize(u_)

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True))
    w_norm = w / sigma

    if update_collection is None:
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
    elif update_collection == "NO_OPS":
        w_norm = tf.reshape(w_norm, w_shape)
    else:
        raise NotImplementedError

    return w_norm


##################################################################################
# Residual Blockes
##################################################################################


def resblock_up(x, out_channels, is_training=True, sn=False, update_collection=None, name="resblock_up"):
    with tf.variable_scope(name):
        x_0 = x
        # block 1
        x = tf.nn.relu(batch_norm(x, is_training=is_training, name="bn1"))
        x = up_sample(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name="conv1")

        # block 2
        x = tf.nn.relu(batch_norm(x, is_training=is_training, name="bn2"))
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name="conv2")

        # skip connection
        x_0 = up_sample(x_0)
        x_0 = conv2d(x_0, out_channels, 1, 1, sn=sn, update_collection=update_collection, name="conv3")

        return x_0 + x


def resblock_down(x, out_channels, sn=False, update_collection=None, downsample=True, name="resblock_down"):
    with tf.variable_scope(name):
        input_channels = x.shape.as_list()[-1]
        x_0 = x
        x = tf.nn.relu(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name="sn_conv1")
        x = tf.nn.relu(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name="sn_conv2")

        if downsample:
            x = down_sample(x)
        if downsample or input_channels != out_channels:
            x_0 = conv2d(x_0, out_channels, 1, 1, sn=sn, update_collection=update_collection, name="sn_conv3")
            if downsample:
                x_0 = down_sample(x_0)

        return x_0 + x


def inblock(x, out_channels, sn=False, update_collection=None, name="inblock"):
    with tf.variable_scope(name):
        x_0 = x
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name="sn_conv1")
        x = tf.nn.relu(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name="sn_conv2")

        x = down_sample(x)
        x_0 = down_sample(x_0)
        x_0 = conv2d(x_0, out_channels, 1, 1, sn=sn, update_collection=update_collection, name="sn_conv3")

        return x_0 + x


##################################################################################
# Loss Functions
##################################################################################


def encoder_gan_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__("wgan"):
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == "dcgan":
        fake_loss = tf.reduce_mean(tf.nn.softplus(-fake))

    if loss_func == "hingegan":
        fake_loss = -tf.reduce_mean(fake)

    return fake_loss
