from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras
import keras.backend as k
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from art.attacks.spatial_transformation import SpatialTransformation
from art.classifiers import KerasClassifier, PyTorchClassifier, TFClassifier
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')
logger.setLevel(10)

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10

W_CONV2D = np.asarray(
    [[[[-0.00789638]], [[-0.00263306]], [[-0.00258223]], [[-0.06708759]], [[0.21071541]], [[0.10087082]],
      [[0.30666843]]],
     [[[0.03002937]], [[0.02245444]], [[-0.07147028]], [[-0.10816725]], [[0.24887177]], [[-0.04827356]],
      [[0.00229796]]],
     [[[-0.18242417]], [[-0.1741664]], [[-0.19573702]], [[-0.14406627]], [[0.08647983]], [[0.20367253]],
      [[-0.09166288]]],
     [[[0.13774922]], [[-0.03155104]], [[-0.1872399]], [[-0.18660806]], [[-0.02223366]], [[0.21571828]],
      [[-0.01274675]]],
     [[[-0.12247422]], [[-0.13241304]], [[0.04481133]], [[0.08032378]], [[-0.05078186]], [[0.04759216]],
      [[-0.14707534]]],
     [[[-0.30721402]], [[-0.05849638]], [[-0.17849001]], [[0.01341257]], [[0.21880499]], [[0.07280245]],
      [[-0.1485026]]],
     [[[-0.03487904]], [[-0.25026447]], [[-0.14562815]], [[0.12643349]], [[-0.19331917]],
      [[-0.24360387]], [[-0.23828903]]]])


def tf_initializer_w_conv2d(shape_list, dtype, partition_info):
    """
    Initializer of weights in convolution layer for Tensorflow.
    :return: Tensorflow constant
    :rtype: tf.constant
    """
    _ = shape_list
    _ = partition_info
    return tf.constant(W_CONV2D, dtype)


def kr_initializer_w_conv2d(_, dtype=None):
    """
        Initializer of weights in convolution layer for Keras.
        :return: Keras variable
        :rtype: k.variable
        """
    return k.variable(value=W_CONV2D, dtype=dtype)


B_CONV2D = np.asarray([0.00311779])


def tf_initializer_b_conv2d(shape_list, dtype, partition_info):
    """
    Initializer of biases in convolution layer for Tensorflow.
    :return: Tensorflow constant
    :rtype: tf.constant
    """
    _ = shape_list
    _ = partition_info
    return tf.constant(B_CONV2D, dtype)


def kr_initializer_b_conv2d(shape, dtype=None):
    """
    Initializer of weights in convolution layer for Keras.
    :return: Keras variable
    :rtype: k.variable
    """
    _ = shape
    return k.variable(value=B_CONV2D, dtype=dtype)


W_DENSE = np.asarray(
    [[-0.13476986, -0.35572886, 0.39324927, -0.22901052, -0.0811693, -0.3123055, -0.15369399, -0.3597307,
      -0.04729861, -0.20822074],
     [-0.0542084, -0.29528973, 0.01068741, -0.15940215, 0.07451159, 0.01864145, 0.13918124, -0.05344852,
      0.18568902, -0.08020786],
     [0.17474903, -0.28545958, 0.1536514, -0.15687232, 0.12206351, -0.31687635, -0.12801449, -0.06631871,
      -0.3750325, -0.23820843],
     [-0.18310688, 0.00591105, -0.43795252, 0.23499434, 0.08124975, 0.0234666, 0.0381292, -0.02565296,
      -0.08529918, -0.05887992],
     [-0.28720114, -0.27321866, 0.0453387, -0.3685731, 0.04172051, -0.19560558, 0.04782663, 0.19110517,
      -0.20616794, -0.05556009],
     [0.08871041, -0.10693074, -0.01452545, 0.52045405, -0.19157553, 0.3104781, -0.13260219, 0.22534075,
      0.00252537, -0.41571805],
     [-0.05230019, 0.10061477, -0.02282099, 0.4105836, 0.1360365, -0.3093183, -0.09113334, -0.1225422,
      0.2608237, -0.11343358],
     [0.10229361, 0.12488309, 0.23242204, -0.2669712, -0.28126365, 0.1460628, -0.2699787, 0.02758999,
      -0.4751196, -0.31990856],
     [0.11416044, -0.12736642, -0.34291124, -0.43725148, 0.54630965, -0.00532502, 0.21765271, 0.16799821,
      -0.33521417, 0.02462992],
     [0.30556166, -0.44640538, 0.38168624, 0.04938773, -0.22284415, -0.22545055, -0.40269086,
      -0.49906304, -0.05910892, 0.06245748],
     [0.23625036, -0.5419616, -0.06358048, -0.33764026, -0.03650349, -0.2699314, -0.58412564, 0.0915556,
      0.3138658, -0.34533662],
     [-0.19475976, -0.3106821, 0.06757818, -0.12326612, 0.10425872, -0.09091422, -0.11154205,
      -0.22407342, -0.21916743, 0.22022846],
     [0.15745547, 0.32471913, 0.01805761, -0.25156206, -0.27749315, -0.25956205, 0.26181027, 0.21421045,
      -0.46093786, -0.0977056],
     [-0.03126516, -0.5134127, 0.36576572, -0.10544802, 0.1555641, -0.02761335, 0.10145999, 0.30376193,
      0.16775244, -0.2708096],
     [-0.00448292, -0.35552138, 0.21349661, -0.14973709, 0.10213355, 0.3771446, -0.11607024, -0.18375315,
      -0.19478855, -0.22998072],
     [-0.02191898, -0.19515613, -0.5287549, -0.37608862, 0.46376917, 0.18349534, -0.41270077, 0.36545402,
      0.12872908, -0.15006325],
     [-0.37827694, -0.39623258, 0.14337724, 0.3938793, 0.24135488, -0.17313641, -0.3206709, -0.3393872,
      -0.2160718, -0.0696595],
     [-0.08790874, 0.09227464, -0.06732312, -0.30081943, 0.19826935, -0.32328498, -0.17068893, 0.3765696,
      -0.3805768, 0.26431862],
     [0.08109575, -0.15298659, -0.4659604, -0.20627183, -0.2854882, -0.05592588, -0.2859281, 0.20579661,
      -0.2124302, -0.09016561],
     [0.34148556, 0.0705517, -0.39427003, -0.29492363, 0.02384043, -0.27866223, -0.05983838, -0.122699,
      -0.3998649, -0.25805104],
     [0.39943328, -0.0086407, 0.06680255, -0.35347003, 0.3737205, -0.19438875, 0.17993195, -0.11919478,
      -0.02604833, 0.02679605],
     [-0.17157973, 0.37023327, -0.39605868, -0.40649268, -0.02218921, -0.13947938, 0.01359389,
      -0.43313634, 0.17709902, -0.15973884],
     [0.13809426, 0.13663352, 0.10038255, 0.01996693, 0.5347731, -0.42451608, 0.21764722, 0.2618781,
      0.27230546, -0.09108792],
     [0.2864959, -0.09312066, 0.28697574, 0.04925587, 0.26510257, -0.3934226, -0.28581426, -0.1151661,
      -0.4022997, -0.12375318],
     [-0.23311849, 0.1678339, 0.5087832, 0.24807404, 0.09471706, 0.1334293, 0.0935249, 0.2915715,
      0.05802418, -0.11327755]])


def tf_initializer_w_dense(shape_list, dtype, partition_info):
    """
    Initializer of weights in dense layer for Tensorflow.
    :return: Tensorflow constant
    :rtype: tf.constant
    """
    _ = shape_list
    _ = partition_info
    return tf.constant(W_DENSE, dtype)


def kr_initializer_w_dense(_, dtype=None):
    """
    Initializer of weights in dense layer for Keras.
    :return: Keras varibale
    :rtype: k.variable
    """
    return k.variable(value=W_DENSE, dtype=dtype)


B_DENSE = np.asarray(
    [-0.08853582, 0.09027167, -0.0260135, 0.01642286, -0.10401244, 0.02632654, -0.00310567, 0.05806893, -0.07686191,
     0.06066821])


def tf_initializer_b_dense(shape_list, dtype, partition_info):
    """
    Initializer of biases in dense layer for Tensorflow.
    :return: Tensorflow constant
    :rtype: tf.constant
    """
    _ = shape_list
    _ = partition_info
    return tf.constant(B_DENSE, dtype)


def kr_initializer_b_dense(_, dtype=None):
    """
    Initializer of biases in dense layer for Keras.
    :return: Keras variable
    :rtype: k.variable
    """
    return k.variable(value=B_DENSE, dtype=dtype)


class Model(nn.Module):
    """
    Create model for pytorch.
    """

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
        w_conv2d_pt = np.swapaxes(W_CONV2D, 0, 2)
        w_conv2d_pt = np.swapaxes(w_conv2d_pt, 1, 3)
        self.conv.weight = nn.Parameter(torch.Tensor(w_conv2d_pt))
        self.conv.bias = nn.Parameter(torch.Tensor(B_CONV2D))
        self.pool = nn.MaxPool2d(4, 4)
        self.fullyconnected = nn.Linear(25, 10)
        self.fullyconnected.weight = nn.Parameter(torch.Tensor(np.transpose(W_DENSE)))
        self.fullyconnected.bias = nn.Parameter(torch.Tensor(B_DENSE))

    def forward(self, x):
        import torch.nn.functional as f

        x = self.pool(f.relu(self.conv(x)))
        x = x.view(-1, 25)
        logit_output = self.fullyconnected(x)

        return logit_output


class TestSpatialTransformation(unittest.TestCase):
    """
    A unittest class for testing Spatial attack.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build a TFClassifier
        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 1, 7, activation=tf.nn.relu, kernel_initializer=tf_initializer_w_conv2d,
                                bias_initializer=tf_initializer_b_conv2d)
        conv = tf.layers.max_pooling2d(conv, 4, 4)
        flattened = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(flattened, 10, kernel_initializer=tf_initializer_w_dense,
                                 bias_initializer=tf_initializer_b_dense)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))

        # Tensorflow session and initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist

        # Train the classifier
        tfc = TFClassifier(clip_values=(0, 1), input_ph=input_ph, logits=logits, output_ph=output_ph, train=None,
                           loss=loss, learning=None, sess=sess)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(tfc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 8, 13, 0] - 0.49004024) <= 0.01)

        self.assertTrue(abs(attack_st.fooling_rate - 0.707) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == 3)
        self.assertTrue(attack_st.attack_trans_y == 3)
        self.assertTrue(attack_st.attack_rot == 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 14, 14, 0] - 0.013572651) <= 0.01)

        sess.close()
        tf.reset_default_graph()

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1),
                         kernel_initializer=kr_initializer_w_conv2d, bias_initializer=kr_initializer_b_conv2d))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax', kernel_initializer=kr_initializer_w_dense,
                        bias_initializer=kr_initializer_b_dense))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        # Get classifier
        krc = KerasClassifier((0, 1), model, use_logits=False)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(krc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 8, 13, 0] - 0.49004024) <= 0.01)
        self.assertTrue(abs(attack_st.fooling_rate - 0.707) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == 3)
        self.assertTrue(attack_st.attack_trans_y == 3)
        self.assertTrue(attack_st.attack_rot == 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 14, 14, 0] - 0.013572651) <= 0.01)

        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Create simple CNN
        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Get classifier
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(ptc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 0, 13, 5] - 0.374206543) <= 0.01)
        self.assertTrue(abs(attack_st.fooling_rate - 0.361) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == 0)
        self.assertTrue(attack_st.attack_trans_y == -3)
        self.assertTrue(attack_st.attack_rot == 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 0, 14, 14] - 0.008591662) <= 0.01)


if __name__ == '__main__':
    unittest.main()
