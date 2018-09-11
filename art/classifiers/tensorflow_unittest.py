from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np
import tensorflow as tf

from art.classifiers.tensorflow import TFImageClassifier, TFTextClassifier
from art.utils import load_mnist, load_imdb
from art.utils import to_categorical


NB_TRAIN = 1000
NB_TEST = 20


# class TestTFImageClassifier(unittest.TestCase):
#     """
#     This class tests the functionalities of the Tensorflow-based classifier.
#     """
#     @classmethod
#     def setUpClass(cls):
#         # Get MNIST
#         (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
#         x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
#         x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
#         cls.mnist = (x_train, y_train), (x_test, y_test)
#
#     def setUp(self):
#         # Define input and output placeholders
#         input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
#         output_ph = tf.placeholder(tf.int32, shape=[None, 10])
#
#         # Define the tensorflow graph
#         conv = tf.layers.conv2d(input_ph, 16, 5, activation=tf.nn.relu)
#         conv = tf.layers.max_pooling2d(conv, 2, 2)
#         fc = tf.contrib.layers.flatten(conv)
#
#         # Logits layer
#         logits = tf.layers.dense(fc, 10)
#
#         # Train operator
#         loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#         train = optimizer.minimize(loss)
#
#         # Tensorflow session and initialization
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#
#         # Create classifier
#         self.classifier = TFImageClassifier((0, 1), input_ph, logits, output_ph, train, loss, None, self.sess)
#
#     def tearDown(self):
#         self.sess.close()
#
#     def test_fit_predict(self):
#         # Get MNIST
#         (x_train, y_train), (x_test, y_test) = self.mnist
#
#         # Test fit and predict
#         self.classifier.fit(x_train, y_train, batch_size=100, nb_epochs=1)
#         preds = self.classifier.predict(x_test)
#         preds_class = np.argmax(preds, axis=1)
#         trues_class = np.argmax(y_test, axis=1)
#         acc = np.sum(preds_class == trues_class) / len(trues_class)
#
#         print("\nAccuracy: %.2f%%" % (acc * 100))
#         self.assertGreater(acc, 0.1)
#         tf.reset_default_graph()
#
#     def test_nb_classes(self):
#         # Start to test
#         self.assertTrue(self.classifier.nb_classes == 10)
#         tf.reset_default_graph()
#
#     def test_input_shape(self):
#         # Start to test
#         self.assertTrue(np.array(self.classifier.input_shape == (28, 28, 1)).all())
#         tf.reset_default_graph()
#
#     def test_class_gradient(self):
#         # Get MNIST
#         (_, _), (x_test, y_test) = self.mnist
#
#         # Test all gradients label = None
#         grads = self.classifier.class_gradient(x_test)
#
#         self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
#         self.assertTrue(np.sum(grads) != 0)
#
#         # Test 1 gradient label = 5
#         grads = self.classifier.class_gradient(x_test, label=5)
#
#         self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
#         self.assertTrue(np.sum(grads) != 0)
#
#         # Test a set of gradients label = array
#         label = np.random.randint(5, size=NB_TEST)
#         grads = self.classifier.class_gradient(x_test, label=label)
#
#         self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
#         self.assertTrue(np.sum(grads) != 0)
#
#         tf.reset_default_graph()
#
#     def test_loss_gradient(self):
#         # Get MNIST
#         (_, _), (x_test, y_test) = self.mnist
#
#         # Test gradient
#         grads = self.classifier.loss_gradient(x_test, y_test)
#
#         self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
#         self.assertTrue(np.sum(grads) != 0)
#         tf.reset_default_graph()
#
#     def test_layers(self):
#         # Get MNIST
#         (_, _), (x_test, y_test) = self.mnist
#
#         # Test and get layers
#         layer_names = self.classifier.layer_names
#         print(layer_names)
#         self.assertTrue(layer_names == ['conv2d/Relu:0', 'max_pooling2d/MaxPool:0',
#                                         'Flatten/flatten/Reshape:0', 'dense/BiasAdd:0'])
#
#         for i, name in enumerate(layer_names):
#             act_i = self.classifier.get_activations(x_test, i)
#             act_name = self.classifier.get_activations(x_test, name)
#             self.assertAlmostEqual(np.sum(act_name - act_i), 0)
#
#         self.assertTrue(self.classifier.get_activations(x_test, 0).shape == (20, 24, 24, 16))
#         self.assertTrue(self.classifier.get_activations(x_test, 1).shape == (20, 12, 12, 16))
#         self.assertTrue(self.classifier.get_activations(x_test, 2).shape == (20, 2304))
#         self.assertTrue(self.classifier.get_activations(x_test, 3).shape == (20, 10))
#         tf.reset_default_graph()


class TestTFTextClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load IMDB
        (x_train, y_train), (x_test, y_test), ids = load_imdb(nb_words=1000, max_length=500)

        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.imdb = (x_train, y_train), (x_test, y_test)
        cls.word_ids = ids

        # Define input and output placeholders
        input_ph = tf.placeholder(tf.int32, shape=[None, 500])
        output_ph = tf.placeholder(tf.int32, shape=[None, 2])

        # Define the tensorflow graph
        embedding_layer = tf.keras.layers.Embedding(1000, 32, input_length=500)(input_ph)
        conv_1d = tf.keras.layers.Conv1D(filters=16, kernel_size=3)(embedding_layer)
        lkrelu1 = tf.keras.layers.LeakyReLU(alpha=.2)(conv_1d)
        mp = tf.keras.layers.MaxPool1D()(lkrelu1)
        flatten = tf.layers.flatten(mp)
        dense = tf.layers.dense(flatten, units=100)
        lkrelu2 = tf.keras.layers.LeakyReLU(alpha=.2)(dense)

        # Logits layer
        logits = tf.layers.dense(lkrelu2, units=2)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        cls.sess = tf.Session()
        cls.sess.run(tf.global_variables_initializer())

        # Create classifier
        cls.classifier = TFTextClassifier(input_ph, logits, embedding_layer, ids, output_ph, train, loss, None,
                                          cls.sess)

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_fit_predict(self):
        (x_train, y_train), (x_test, y_test) = self.imdb
        y_train = to_categorical(y_train, nb_classes=2)
        y_test = to_categorical(y_test, nb_classes=2)

        acc1 = np.sum(np.argmax(self.classifier.predict(x_test), axis=1) == y_test) / x_test.shape[0]
        print('\nAccuracy: %.2f%%' % (acc1 * 100))

        self.classifier.fit(x_train, y_train, nb_epochs=1, batch_size=10)
        acc2 = np.sum(np.argmax(self.classifier.predict(x_test), axis=1) == y_test) / x_test.shape[0]
        print("\nAccuracy: %.2f%%" % (acc2 * 100))

        self.assertTrue(acc2 >= acc1)

    def test_nb_classes(self):
        # Start to test
        self.assertTrue(self.classifier.nb_classes == 2)
        tf.reset_default_graph()

    def test_input_shape(self):
        # Start to test
        self.assertTrue(np.array(self.classifier.input_shape == (500)).all())
        tf.reset_default_graph()

    def test_class_gradient(self):
        # Get IMDB
        (_, _), (x_test, y_test) = self.imdb
        y_test = to_categorical(y_test, nb_classes=2)

        # Test all gradients label = None
        grads = self.classifier.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 2, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test 1 gradient label = 5
        grads = self.classifier.class_gradient(x_test, label=1)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test a set of gradients label = array
        label = np.random.randint(2, size=NB_TEST)
        grads = self.classifier.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)

        tf.reset_default_graph()

    def test_loss_gradient(self):
        # Get IMDB
        (_, _), (x_test, y_test) = self.imdb
        y_test = to_categorical(y_test, nb_classes=2)

        # Test gradient
        grads = self.classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)
        tf.reset_default_graph()

    def test_layers(self):
        # Get IMDB
        (_, _), (x_test, y_test) = self.imdb
        y_test = to_categorical(y_test, nb_classes=2)

        # Test and get layers
        layer_names = self.classifier.layer_names
        print(layer_names)
        self.assertTrue(layer_names == ['conv2d/Relu:0', 'max_pooling2d/MaxPool:0',
                                        'Flatten/flatten/Reshape:0', 'dense/BiasAdd:0'])

        for i, name in enumerate(layer_names):
            act_i = self.classifier.get_activations(x_test, i)
            act_name = self.classifier.get_activations(x_test, name)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)

        print(self.classifier.get_activations(x_test, 0).shape)
        print(self.classifier.get_activations(x_test, 1).shape)
        print(self.classifier.get_activations(x_test, 2).shape)
        print(self.classifier.get_activations(x_test, 3).shape)
        print(self.classifier.get_activations(x_test, 4).shape)
        print(self.classifier.get_activations(x_test, 5).shape)
        print(self.classifier.get_activations(x_test, 6).shape)
        print(self.classifier.get_activations(x_test, 7).shape)
        self.assertTrue(self.classifier.get_activations(x_test, 0).shape == (20, 24, 24, 16))
        self.assertTrue(self.classifier.get_activations(x_test, 1).shape == (20, 12, 12, 16))
        self.assertTrue(self.classifier.get_activations(x_test, 2).shape == (20, 2304))
        self.assertTrue(self.classifier.get_activations(x_test, 3).shape == (20, 10))
        tf.reset_default_graph()


    def test_embedding(self):
        return
        # (x_train, y_train), (x_test, y_test) = self.imdb
        #
        # classifier = KerasTextClassifier(model=self.model, loss=k.binary_crossentropy, use_logits=False)


if __name__ == '__main__':
    unittest.main()



