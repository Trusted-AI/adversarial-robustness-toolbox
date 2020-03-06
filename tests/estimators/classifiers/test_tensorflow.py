# MIT License
#
# Copyright (C) IBM Corporation 2018
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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import tensorflow as tf
import numpy as np

from art.data_generators import TFDataGenerator

from tests.utils import TestBase, master_seed, get_classifier_tf

logger = logging.getLogger(__name__)


class TestTensorFlowClassifier(TestBase):
    """
    This class tests the TensorFlow classifier.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.classifier, cls.sess = get_classifier_tf()
        cls.classifier_logits, _ = get_classifier_tf(from_logits=True)

        if tf.__version__[0] == '2':
            cls.is_version_2 = True
        else:
            cls.is_version_2 = False

    def setUp(self):
        super().setUp()

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_mnist[0:1])
        y_expected = np.asarray([[0.12109935, 0.0498215, 0.0993958, 0.06410097, 0.11366927, 0.04645343, 0.06419806,
                                  0.30685693, 0.07616713, 0.05823758]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_fit_generator(self):
        if not self.is_version_2:
            classifier, sess = get_classifier_tf()

            # Create TensorFlow data generator
            x_tensor = tf.convert_to_tensor(self.x_train_mnist.reshape(10, 100, 28, 28, 1))
            y_tensor = tf.convert_to_tensor(self.y_train_mnist.reshape(10, 100, 10))
            dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
            iterator = dataset.make_initializable_iterator()
            data_gen = TFDataGenerator(sess=sess, iterator=iterator, iterator_type='initializable', iterator_arg={},
                                       size=1000, batch_size=100)

            # Test fit and predict
            classifier.fit_generator(data_gen, nb_epochs=2)
            predictions = classifier.predict(self.x_test_mnist)
            predictions_class = np.argmax(predictions, axis=1)
            true_class = np.argmax(self.y_test_mnist, axis=1)
            accuracy = np.sum(predictions_class == true_class) / len(true_class)

            logger.info('Accuracy after fitting TensorFlow classifier with generator: %.2f%%', (accuracy * 100))
            self.assertAlmostEqual(accuracy, 0.65, delta=0.02)

    def test_nb_classes(self):
        self.assertEqual(self.classifier.nb_classes, 10)

    def test_input_shape(self):
        self.assertEqual(self.classifier.input_shape, (28, 28, 1))

    def test_class_gradient(self):

        # Test all gradients label = None
        gradients = self.classifier_logits.class_gradient(self.x_test_mnist)

        self.assertEqual(gradients.shape, (self.n_test, 10, 28, 28, 1))

        expected_gradients_1 = np.asarray([-0.03347399, -0.03195872, -0.02650188, 0.04111874, 0.08676253, 0.03339913,
                                           0.06925241, 0.09387045, 0.15184258, -0.00684002, 0.05070481, 0.01409407,
                                           -0.03632583, 0.00151133, 0.05102589, 0.00766463, -0.00898967, 0.00232938,
                                           -0.00617045, -0.00201032, 0.00410065, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 5, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.09723657, -0.00240533, 0.02445251, -0.00035474, 0.04765627, 0.04286841,
                                           0.07209076, 0.0, 0.0, -0.07938144, -0.00142567, 0.02882954,
                                           -0.00049514, 0.04170151, 0.05102589, 0.09544909, -0.04401167, -0.06158172,
                                           0.03359772, -0.00838454, 0.01722163, -0.13376027, 0.08206709, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 5, :, 14, 0], expected_gradients_2, decimal=4)

        # Test 1 gradient label = 5
        gradients = self.classifier_logits.class_gradient(self.x_test_mnist, label=5)

        self.assertEqual(gradients.shape, (self.n_test, 1, 28, 28, 1))

        expected_gradients_1 = np.asarray([-0.03347399, -0.03195872, -0.02650188, 0.04111874, 0.08676253, 0.03339913,
                                           0.06925241, 0.09387045, 0.15184258, -0.00684002, 0.05070481, 0.01409407,
                                           -0.03632583, 0.00151133, 0.05102589, 0.00766463, -0.00898967, 0.00232938,
                                           -0.00617045, -0.00201032, 0.00410065, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.09723657, -0.00240533, 0.02445251, -0.00035474, 0.04765627, 0.04286841,
                                           0.07209076, 0.0, 0.0, -0.07938144, -0.00142567, 0.02882954,
                                           -0.00049514, 0.04170151, 0.05102589, 0.09544909, -0.04401167, -0.06158172,
                                           0.03359772, -0.00838454, 0.01722163, -0.13376027, 0.08206709, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=self.n_test)
        gradients = self.classifier_logits.class_gradient(self.x_test_mnist, label=label)

        self.assertEqual(gradients.shape, (self.n_test, 1, 28, 28, 1))

        expected_gradients_1 = np.asarray([0.06860766, 0.065502, 0.08539103, 0.13868105, -0.05520725, -0.18788849,
                                           0.02264893, 0.02980516, 0.2226511, 0.11288887, -0.00678776, 0.02045561,
                                           -0.03120914, 0.00642691, 0.08449504, 0.02848018, -0.03251382, 0.00854315,
                                           -0.02354656, -0.00767687, 0.01565931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.0487146, -0.0171556, -0.03161772, -0.0420007, 0.03360246, -0.01864819,
                                           0.00315916, 0.0, 0.0, -0.07631349, -0.00374462, 0.04229517,
                                           -0.01131879, 0.05044588, 0.08449504, 0.12417868, 0.07536847, 0.03906382,
                                           0.09467953, 0.00543209, -0.00504872, -0.03366479, -0.00385999, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)

    def test_loss_gradient(self):

        gradients = self.classifier.loss_gradient(self.x_test_mnist, self.y_test_mnist)

        self.assertEqual(gradients.shape, (self.n_test, 28, 28, 1))

        expected_gradients_1 = np.asarray([5.59206062e-04, 5.33892540e-04, 6.48919027e-04, 7.92516454e-04,
                                           -4.02929145e-04, -1.12814642e-03, 1.85060024e-04, 3.25053406e-05,
                                           8.16319487e-04, 3.33394884e-04, 3.17659928e-04, -2.42046357e-04,
                                           -7.81555660e-04, -4.69873514e-04, 1.07115903e-03, 4.08643362e-04,
                                           -3.44107364e-04, 1.07128391e-04, -4.22919547e-04, -1.38615724e-04,
                                           2.82748661e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([2.10802755e-05, 2.13919120e-05, 5.20980720e-05, 5.48000680e-05,
                                           -2.30590031e-05, 4.32076595e-05, 2.74944887e-05, 0.00000000e+00,
                                           0.00000000e+00, -5.83440997e-04, -6.16604229e-05, 5.26219024e-04,
                                           -2.37398461e-04, 5.27310593e-04, 1.07115903e-03, 1.27738668e-03,
                                           6.89289009e-04, 1.33779933e-04, 1.00320193e-03, 1.68109560e-04,
                                           -2.86467184e-06, -5.58885862e-04, 1.47416518e-04, 0.00000000e+00,
                                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, :, 14, 0], expected_gradients_2, decimal=4)

    def test_layers(self):
        if not self.is_version_2:
            layer_names = self.classifier.layer_names

            for i, name in enumerate(layer_names):
                activation_i = self.classifier.get_activations(self.x_test_mnist, i, batch_size=5)
                activation_name = self.classifier.get_activations(self.x_test_mnist, name, batch_size=5)
                np.testing.assert_array_equal(activation_name, activation_i)

    # Commented because of problems with multiple classifiers in the same test module
    # def test_save(self):
    #     if not self.is_version_2:
    #         path = 'tmp'
    #         filename = 'model.ckpt'
    #
    #         # Save
    #         self.classifier.save(filename, path=path)
    #         self.assertTrue(os.path.isfile(os.path.join(path, filename, 'variables/variables.data-00000-of-00001')))
    #         self.assertTrue(os.path.isfile(os.path.join(path, filename, 'variables/variables.index')))
    #         self.assertTrue(os.path.isfile(os.path.join(path, filename, 'saved_model.pb')))
    #
    #         # # Restore
    #         # with tf.Session(graph=tf.Graph()) as sess:
    #         #     tf.saved_model.loader.load(sess, ["serve"], os.path.join(path, filename))
    #         #     graph = tf.get_default_graph()
    #         #     sess.run('SavedOutput:0', feed_dict={'SavedInputPhD:0': input_batch})
    #
    #         # Remove saved files
    #         shutil.rmtree(os.path.join(path, filename))

    def test_set_learning(self):
        if not self.is_version_2:
            self.assertEqual(self.classifier._feed_dict, {})
            self.classifier.set_learning_phase(False)
            self.assertFalse(self.classifier._feed_dict[self.classifier._learning])
            self.classifier.set_learning_phase(True)
            self.assertTrue(self.classifier._feed_dict[self.classifier._learning])
            self.assertTrue(self.classifier.learning_phase)

    def test_repr(self):

        repr_classifier = repr(self.classifier)

        if self.is_version_2:
            self.assertIn('TensorFlowV2Classifier', repr_classifier)
            self.assertIn('model=', repr_classifier)
            self.assertIn('nb_classes=10', repr_classifier)
            self.assertIn('input_shape=(28, 28, 1)', repr_classifier)
            self.assertIn('loss_object=<tensorflow.python.keras.losses.SparseCategoricalCrossentropy', repr_classifier)
            self.assertIn('train_step=<function get_classifier_tf_v2.<locals>.train_step', repr_classifier)
        else:
            self.assertIn('TensorFlowClassifier', repr_classifier)
            self.assertIn('input_ph=<tf.Tensor \'Placeholder:0\' shape=(?, 28, 28, 1) dtype=float32>', repr_classifier)
            self.assertIn('output=<tf.Tensor \'Softmax:0\' shape=(?, 10) dtype=float32>', repr_classifier)
            self.assertIn('labels_ph=<tf.Tensor \'Placeholder_1:0\' shape=(?, 10) dtype=int32>', repr_classifier)
            self.assertIn('train=<tf.Operation \'Adam\' type=NoOp>', repr_classifier)
            self.assertIn('loss=<tf.Tensor \'Mean:0\' shape=() dtype=float32>', repr_classifier)
            self.assertIn('learning=None', repr_classifier)
            self.assertIn('sess=<tensorflow.python.client.session.Session object', repr_classifier)
            self.assertIn('TensorFlowClassifier', repr_classifier)

        self.assertIn('channel_index=3, clip_values=(0, 1), preprocessing_defences=None, postprocessing_defences=None, '
                      'preprocessing=(0, 1))', repr_classifier)

    # Commented because of problems with multiple classifiers in the same test module
    # def test_pickle(self):
    #     if not self.is_version_2:
    #         classifier = self.classifier
    #
    #         full_path = os.path.join(ART_DATA_PATH, 'my_classifier')
    #         folder = os.path.split(full_path)[0]
    #
    #         if not os.path.exists(folder):
    #             os.makedirs(folder)
    #
    #         pickle.dump(classifier, open(full_path, 'wb'))
    #
    #         # Unpickle:
    #         with open(full_path, 'rb') as f:
    #             classifier_loaded = pickle.load(f)
    #             self.assertEqual(classifier._clip_values, classifier_loaded._clip_values)
    #             self.assertEqual(classifier._channel_index, classifier_loaded._channel_index)
    #             self.assertEqual(set(classifier.__dict__.keys()), set(classifier_loaded.__dict__.keys()))
    #
    #         # Test predict
    #         predictions_1 = classifier.predict(self.x_test)
    #         accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
    #         predictions_2 = classifier_loaded.predict(self.x_test)
    #         accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
    #         self.assertEqual(accuracy_1, accuracy_2)


if __name__ == '__main__':
    unittest.main()
