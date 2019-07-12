from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
import tensorflow as tf

from art.utils import load_mnist, master_seed
from art.utils_test import get_classifier_tf

logger = logging.getLogger('testLogger')

NB_TRAIN = 1000
NB_TEST = 20


class TestTFClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the Tensorflow-based classifier.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        cls.classifier, cls.sess = get_classifier_tf()

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()
        cls.sess.close()

    def test_predict(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test fit and predict
        preds = self.classifier.predict(x_test)
        preds_class = np.argmax(preds, axis=1)
        trues_class = np.argmax(y_test, axis=1)
        acc = np.sum(preds_class == trues_class) / len(trues_class)

        logger.info('Accuracy after fitting: %.2f%%', (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_fit_generator(self):
        from art.data_generators import TFDataGenerator

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Create Tensorflow data generator
        x_tensor = tf.convert_to_tensor(x_train.reshape(10, 100, 28, 28, 1))
        y_tensor = tf.convert_to_tensor(y_train.reshape(10, 100, 10))
        dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
        iterator = dataset.make_initializable_iterator()
        data_gen = TFDataGenerator(sess=self.sess, iterator=iterator, iterator_type='initializable',
                                   iterator_arg={}, size=1000, batch_size=100)

        # Test fit and predict
        self.classifier.fit_generator(data_gen, nb_epochs=2)
        preds = self.classifier.predict(x_test)
        preds_class = np.argmax(preds, axis=1)
        trues_class = np.argmax(y_test, axis=1)
        acc = np.sum(preds_class == trues_class) / len(trues_class)

        logger.info('Accuracy after fitting TF classifier with generator: %.2f%%', (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        # Start to test
        self.assertEqual(self.classifier.nb_classes, 10)

    def test_input_shape(self):
        # Start to test
        self.assertTrue(np.array(self.classifier.input_shape == (28, 28, 1)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test all gradients label = None
        grads = self.classifier.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test 1 gradient label = 5
        grads = self.classifier.class_gradient(x_test, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = self.classifier.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test gradient
        grads = self.classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test and get layers
        layer_names = self.classifier.layer_names
        logger.debug(layer_names)

        for i, name in enumerate(layer_names):
            act_i = self.classifier.get_activations(x_test, i, batch_size=5)
            act_name = self.classifier.get_activations(x_test, name, batch_size=5)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)

    def test_save(self):
        import os
        import shutil

        path = 'tmp'
        filename = 'model.ckpt'

        # Save
        self.classifier.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename, 'variables/variables.data-00000-of-00001')))
        self.assertTrue(os.path.isfile(os.path.join(path, filename, 'variables/variables.index')))
        self.assertTrue(os.path.isfile(os.path.join(path, filename, 'saved_model.pb')))

        # # Restore
        # with tf.Session(graph=tf.Graph()) as sess:
        #     tf.saved_model.loader.load(sess, ["serve"], os.path.join(path, filename))
        #     graph = tf.get_default_graph()
        #     print(graph.get_operations())
        #     sess.run('SavedOutputLogit:0', feed_dict={'SavedInputPhD:0': input_batch})

        # Remove saved files
        shutil.rmtree(os.path.join(path, filename))

    def test_set_learning(self):
        tfc = self.classifier

        self.assertEqual(tfc._feed_dict, {})
        tfc.set_learning_phase(False)
        self.assertFalse(tfc._feed_dict[tfc._learning])
        tfc.set_learning_phase(True)
        self.assertTrue(tfc._feed_dict[tfc._learning])
        self.assertTrue(tfc.learning_phase)

    def test_repr(self):
        repr_ = repr(self.classifier)
        self.assertIn('art.classifiers.tensorflow.TFClassifier', repr_)
        self.assertIn('channel_index=3, clip_values=(0, 1)', repr_)
        self.assertIn('defences=None, preprocessing=(0, 1)', repr_)

    def test_pickle(self):
        import os
        from art import DATA_PATH
        full_path = os.path.join(DATA_PATH, 'my_classifier')
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        import pickle
        pickle.dump(self.classifier, open(full_path, 'wb'))

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Unpickle:
        with open(full_path, 'rb') as f:
            loaded = pickle.load(f)
            self.assertEqual(self.classifier._clip_values, loaded._clip_values)
            self.assertEqual(self.classifier._channel_index, loaded._channel_index)
            self.assertEqual(set(self.classifier.__dict__.keys()), set(loaded.__dict__.keys()))

        # Test predict
        preds1 = self.classifier.predict(x_test)
        acc1 = np.sum(np.argmax(preds1, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        preds2 = loaded.predict(x_test)
        acc2 = np.sum(np.argmax(preds2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        self.assertEqual(acc1, acc2)

        loaded._sess.close()


if __name__ == '__main__':
    unittest.main()
