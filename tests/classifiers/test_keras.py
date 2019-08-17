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

import os
import logging
import unittest
import requests
import tempfile
import shutil
import pickle

import numpy as np
import keras
import keras.backend as k
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

from art import DATA_PATH
from art.classifiers import KerasClassifier
from art.classifiers.keras import generator_fit
from art.defences import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.utils import load_mnist, master_seed
from art.utils_test import get_classifier_kr
from art.data_generators import KerasDataGenerator

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestKerasClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.clear_session()
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Load small Keras model
        cls.functional_model = cls.functional_model()

        # Temporary folder for tests
        cls.test_dir = tempfile.mkdtemp()

        # Download one ImageNet pic for tests
        url = 'http://farm1.static.flickr.com/163/381342603_81db58bea4.jpg'
        result = requests.get(url, stream=True)
        if result.status_code == 200:
            image = result.raw.read()
            f = open(os.path.join(cls.test_dir, 'test.jpg'), 'wb')
            f.write(image)
            f.close()

    @classmethod
    def tearDownClass(cls):
        k.clear_session()
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @staticmethod
    def functional_model():
        in_layer = Input(shape=(28, 28, 1), name="input0")
        layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(in_layer)
        layer = Conv2D(64, (3, 3), activation='relu')(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)
        layer = Dropout(0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation='relu')(layer)
        layer = Dropout(0.5)(layer)
        out_layer = Dense(10, activation='softmax', name="output0")(layer)

        in_layer_2 = Input(shape=(28, 28, 1), name="input1")
        layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(in_layer_2)
        layer = Conv2D(64, (3, 3), activation='relu')(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)
        layer = Dropout(0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation='relu')(layer)
        layer = Dropout(0.5)(layer)
        out_layer_2 = Dense(10, activation='softmax', name="output1")(layer)

        model = Model(inputs=[in_layer, in_layer_2], outputs=[out_layer, out_layer_2])

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'], loss_weights=[1.0, 1.0])

        return model

    def test_fit(self):
        labels = np.argmax(self.mnist[1][1], axis=1)
        classifier = get_classifier_kr()
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc2 * 100))

        self.assertEqual(acc, 0.32)
        self.assertEqual(acc2, 0.73)

    def test_fit_generator(self):
        labels = np.argmax(self.mnist[1][1], axis=1)
        classifier = get_classifier_kr()
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        gen = generator_fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE)
        data_gen = KerasDataGenerator(generator=gen, size=NB_TRAIN, batch_size=BATCH_SIZE)
        classifier.fit_generator(generator=data_gen, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc2 * 100))

        self.assertEqual(acc, 0.32)
        self.assertEqual(acc2, 0.36)

    def test_fit_image_generator(self):
        x_train, y_train = self.mnist[0]
        labels_test = np.argmax(self.mnist[1][1], axis=1)
        classifier = get_classifier_kr()
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels_test) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        keras_gen = ImageDataGenerator(width_shift_range=0.075, height_shift_range=0.075, rotation_range=12,
                                       shear_range=0.075, zoom_range=0.05, fill_mode='constant', cval=0)
        keras_gen.fit(x_train)
        data_gen = KerasDataGenerator(generator=keras_gen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                                      size=NB_TRAIN, batch_size=BATCH_SIZE)
        classifier.fit_generator(generator=data_gen, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels_test) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc2 * 100))

        self.assertEqual(acc, 0.32)
        self.assertAlmostEqual(acc2, 0.35, delta=0.02)

    def test_fit_kwargs(self):

        def get_lr(_):
            return 0.01

        # Test a valid callback
        classifier = get_classifier_kr()
        kwargs = {'callbacks': [LearningRateScheduler(get_lr)]}
        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=1, **kwargs)

        # Test failure for invalid parameters
        kwargs = {'epochs': 1}
        with self.assertRaises(TypeError) as context:
            classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=1, **kwargs)

        self.assertIn('multiple values for keyword argument', str(context.exception))

    def test_shapes(self):
        x_test, y_test = self.mnist[1]
        classifier = get_classifier_kr()

        preds = classifier.predict(self.mnist[1][0])
        self.assertEqual(preds.shape, y_test.shape)

        self.assertEqual(classifier.nb_classes, 10)

        class_grads = classifier.class_gradient(x_test[:11])
        self.assertEqual(class_grads.shape, tuple([11, 10] + list(x_test[1].shape)))

        loss_grads = classifier.loss_gradient(x_test[:11], y_test[:11])
        self.assertEqual(loss_grads.shape, x_test[:11].shape)

    def test_defences_predict(self):

        (_, _), (x_test, y_test) = self.mnist

        clip_values = (0, 1)
        fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
        jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
        smooth = SpatialSmoothing()
        classifier_ = get_classifier_kr()
        classifier = KerasClassifier(clip_values=clip_values, model=classifier_._model, defences=[fs, jpeg, smooth])
        self.assertEqual(len(classifier.defences), 3)

        preds_classifier = classifier.predict(x_test)

        # Apply the same defences by hand
        x_test_defense = x_test
        x_test_defense, _ = fs(x_test_defense, y_test)
        x_test_defense, _ = jpeg(x_test_defense, y_test)
        x_test_defense, _ = smooth(x_test_defense, y_test)
        classifier = get_classifier_kr()
        preds_check = classifier._model.predict(x_test_defense)

        # Check that the prediction results match
        np.testing.assert_array_almost_equal(preds_classifier, preds_check, decimal=4)

    def test_class_gradient(self):
        (_, _), (x_test, _) = self.mnist
        classifier = get_classifier_kr()

        # Test all gradients label
        grads = classifier.class_gradient(x_test)

        self.assertTrue(grads.shape == (NB_TEST, 10, 28, 28, 1))

        expected_grads_1 = np.asarray([-1.0557447e-03, -1.0079544e-03, -7.7426434e-04, 1.7387432e-03,
                                       2.1773507e-03, 5.0880699e-05, 1.6497371e-03, 2.6113100e-03,
                                       6.0904310e-03, 4.1080985e-04, 2.5268078e-03, -3.6661502e-04,
                                       -3.0568996e-03, -1.1665225e-03, 3.8904310e-03, 3.1726385e-04,
                                       1.3203260e-03, -1.1720930e-04, -1.4315104e-03, -4.7676818e-04,
                                       9.7251288e-04, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(grads[0, 5, 14, :, 0], expected_grads_1, decimal=4)

        expected_grads_2 = np.asarray([-0.00367321, -0.0002892, 0.00037825, -0.00053344, 0.00192121, 0.00112047,
                                       0.0023135, 0.0, 0.0, -0.00391743, -0.0002264, 0.00238103,
                                       -0.00073711, 0.00270405, 0.00389043, 0.00440818, -0.00412769, -0.00441794,
                                       0.00081916, -0.00091284, 0.00119645, -0.00849089, 0.00547925, 0.0,
                                       0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(grads[0, 5, :, 14, 0], expected_grads_2, decimal=4)

        # Test 1 gradient label = 5
        grads = classifier.class_gradient(x_test, label=5)

        self.assertTrue(grads.shape == (NB_TEST, 1, 28, 28, 1))

        expected_grads_1 = np.asarray([-1.0557447e-03, -1.0079544e-03, -7.7426434e-04, 1.7387432e-03,
                                       2.1773507e-03, 5.0880699e-05, 1.6497371e-03, 2.6113100e-03,
                                       6.0904310e-03, 4.1080985e-04, 2.5268078e-03, -3.6661502e-04,
                                       -3.0568996e-03, -1.1665225e-03, 3.8904310e-03, 3.1726385e-04,
                                       1.3203260e-03, -1.1720930e-04, -1.4315104e-03, -4.7676818e-04,
                                       9.7251288e-04, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(grads[0, 0, 14, :, 0], expected_grads_1, decimal=4)

        expected_grads_2 = np.asarray([-0.00367321, -0.0002892, 0.00037825, -0.00053344, 0.00192121, 0.00112047,
                                       0.0023135, 0.0, 0.0, -0.00391743, -0.0002264, 0.00238103,
                                       -0.00073711, 0.00270405, 0.00389043, 0.00440818, -0.00412769, -0.00441794,
                                       0.00081916, -0.00091284, 0.00119645, -0.00849089, 0.00547925, 0.0,
                                       0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(grads[0, 0, :, 14, 0], expected_grads_2, decimal=4)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = classifier.class_gradient(x_test, label=label)

        self.assertTrue(grads.shape == (NB_TEST, 1, 28, 28, 1))

        expected_grads_1 = np.asarray([5.0867125e-03, 4.8564528e-03, 6.1040390e-03, 8.6531248e-03,
                                       -6.0958797e-03, -1.4114540e-02, -7.1085989e-04, -5.0330814e-04,
                                       1.2943064e-02, 8.2416134e-03, -1.9859476e-04, -9.8109958e-05,
                                       -3.8902222e-03, -1.2945873e-03, 7.5137997e-03, 1.7720886e-03,
                                       3.1399424e-04, 2.3657181e-04, -3.0891625e-03, -1.0211229e-03,
                                       2.0828887e-03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(grads[0, 0, 14, :, 0], expected_grads_1, decimal=4)

        expected_grads_2 = np.asarray([-0.00195835, -0.00134457, -0.00307221, -0.00340564, 0.00175022, -0.00239714,
                                       -0.00122619, 0.0, 0.0, -0.00520899, -0.00046105, 0.00414874,
                                       -0.00171095, 0.00429184, 0.0075138, 0.00792442, 0.0019566, 0.00035517,
                                       0.00504575, -0.00037397, 0.00022343, -0.00530034, 0.0020528, 0.0,
                                       0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(grads[0, 0, :, 14, 0], expected_grads_2, decimal=4)

    def test_loss_gradient(self):
        (_, _), (x_test, y_test) = self.mnist
        classifier = get_classifier_kr()

        # Test gradient
        grads = classifier.loss_gradient(x_test, y_test)

        self.assertTrue(grads.shape == (NB_TEST, 28, 28, 1))

        expected_grads_1 = np.asarray([0.0559206, 0.05338925, 0.0648919, 0.07925165, -0.04029291, -0.11281465,
                                       0.01850601, 0.00325054, 0.08163195, 0.03333949, 0.031766, -0.02420463,
                                       -0.07815556, -0.04698735, 0.10711591, 0.04086434, -0.03441073, 0.01071284,
                                       -0.04229195, -0.01386157, 0.02827487, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(grads[0, 14, :, 0], expected_grads_1, decimal=4)

        expected_grads_2 = np.asarray([0.00210803, 0.00213919, 0.00520981, 0.00548001, -0.0023059, 0.00432077,
                                       0.00274945, 0.0, 0.0, -0.0583441, -0.00616604, 0.0526219,
                                       -0.02373985, 0.05273106, 0.10711591, 0.12773865, 0.0689289, 0.01337799,
                                       0.10032021, 0.01681096, -0.00028647, -0.05588859, 0.01474165, 0.0,
                                       0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(grads[0, :, 14, 0], expected_grads_2, decimal=4)

    def test_functional_model(self):
        # Need to update the functional_model code to produce a model with more than one input and output layers...
        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=1, output_layer=1)
        self.assertTrue(keras_model._input.name, "input1")
        self.assertTrue(keras_model._output.name, "output1")

        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=0, output_layer=0)
        self.assertTrue(keras_model._input.name, "input0")
        self.assertTrue(keras_model._output.name, "output0")

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]

        classifier = get_classifier_kr()
        self.assertEqual(len(classifier.layer_names), 3)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            act_i = classifier.get_activations(x_test, i)
            act_name = classifier.get_activations(x_test, name)
            np.testing.assert_array_equal(act_name, act_i)

    def test_resnet(self):
        keras.backend.set_learning_phase(0)
        model = ResNet50(weights='imagenet')
        classifier = KerasClassifier(model, clip_values=(0, 255))

        # Load image from file
        image = img_to_array(load_img(os.path.join(self.test_dir, 'test.jpg'), target_size=(224, 224)))
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        prediction = classifier.predict(image)
        label = decode_predictions(prediction)[0][0]

        self.assertEqual(label[1], 'Weimaraner')
        self.assertAlmostEqual(prediction[0, 178], 0.2658045, places=3)

    def test_learning_phase(self):
        classifier = get_classifier_kr()

        self.assertFalse(hasattr(classifier, '_learning_phase'))
        classifier.set_learning_phase(False)
        self.assertFalse(classifier.learning_phase)
        classifier.set_learning_phase(True)
        self.assertTrue(classifier.learning_phase)
        self.assertTrue(hasattr(classifier, '_learning_phase'))

    def test_save(self):
        path = 'tmp'
        filename = 'model.h5'
        classifier = get_classifier_kr()
        classifier.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename)))

        # Remove saved file
        os.remove(os.path.join(path, filename))

    def test_pickle(self):
        filename = 'my_classifier.p'
        full_path = os.path.join(DATA_PATH, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=1, output_layer=1,
                                      defences=fs)
        with open(full_path, 'wb') as save_file:
            pickle.dump(keras_model, save_file)

        # Unpickle:
        with open(full_path, 'rb') as load_file:
            loaded = pickle.load(load_file)

            self.assertEqual(keras_model._clip_values, loaded._clip_values)
            self.assertEqual(keras_model._channel_index, loaded._channel_index)
            self.assertEqual(keras_model._use_logits, loaded._use_logits)
            self.assertEqual(keras_model._input_layer, loaded._input_layer)
            self.assertEqual(self.functional_model.get_config(), loaded._model.get_config())
            self.assertTrue(isinstance(loaded.defences[0], FeatureSqueezing))

        os.remove(full_path)

    def test_repr(self):
        classifier = get_classifier_kr()
        repr_ = repr(classifier)
        self.assertIn('art.classifiers.keras.KerasClassifier', repr_)
        self.assertIn('use_logits=False, channel_index=3', repr_)
        self.assertIn('clip_values=(0, 1), defences=None, preprocessing=(0, 1)', repr_)
        self.assertIn('input_layer=0, output_layer=0', repr_)
