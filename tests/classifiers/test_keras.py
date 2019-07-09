from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras
import keras.backend as k
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten
from keras.models import Model

from art.classifiers import KerasClassifier
from art.defences import FeatureSqueezing
from art.utils import load_mnist, master_seed
from art.utils_test import get_classifier_kr

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
        cls.model_mnist, _ = get_classifier_kr()
        cls.functional_model = cls.functional_model()

        import requests
        import tempfile
        import os

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

        import shutil
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
        classifier = self.model_mnist
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc2 * 100))

        self.assertGreaterEqual(acc2, 0.9 * acc)

    def test_fit_generator(self):
        from art.classifiers.keras import generator_fit
        from art.data_generators import KerasDataGenerator

        labels = np.argmax(self.mnist[1][1], axis=1)
        classifier = self.model_mnist
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        gen = generator_fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE)
        data_gen = KerasDataGenerator(generator=gen, size=NB_TRAIN, batch_size=BATCH_SIZE)
        classifier.fit_generator(generator=data_gen, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc2 * 100))

        self.assertGreaterEqual(acc2, 0.8 * acc)

    def test_fit_image_generator(self):
        from keras.preprocessing.image import ImageDataGenerator
        from art.data_generators import KerasDataGenerator

        x_train, y_train = self.mnist[0]
        labels_test = np.argmax(self.mnist[1][1], axis=1)
        classifier = self.model_mnist
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

        self.assertGreaterEqual(acc2, 0.8 * acc)

    def test_fit_kwargs(self):
        from keras.callbacks import LearningRateScheduler

        def get_lr(_):
            return 0.01

        # Test a valid callback
        classifier = self.model_mnist
        kwargs = {'callbacks': [LearningRateScheduler(get_lr)]}
        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=1, **kwargs)

        # Test failure for invalid parameters
        kwargs = {'epochs': 1}
        with self.assertRaises(TypeError) as context:
            classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=1, **kwargs)

        self.assertIn('multiple values for keyword argument', str(context.exception))

    def test_shapes(self):
        x_test, y_test = self.mnist[1]
        classifier = self.model_mnist

        preds = classifier.predict(self.mnist[1][0])
        self.assertEqual(preds.shape, y_test.shape)

        self.assertEqual(classifier.nb_classes, 10)

        class_grads = classifier.class_gradient(x_test[:11])
        self.assertEqual(class_grads.shape, tuple([11, 10] + list(x_test[1].shape)))

        loss_grads = classifier.loss_gradient(x_test[:11], y_test[:11])
        self.assertEqual(loss_grads.shape, x_test[:11].shape)

    def test_defences_predict(self):
        from art.defences import FeatureSqueezing, JpegCompression, SpatialSmoothing

        (_, _), (x_test, y_test) = self.mnist

        clip_values = (0, 1)
        fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
        jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
        smooth = SpatialSmoothing()
        classifier = KerasClassifier(clip_values=clip_values, model=self.model_mnist._model,
                                     defences=[fs, jpeg, smooth])
        self.assertEqual(len(classifier.defences), 3)

        preds_classifier = classifier.predict(x_test)

        # Apply the same defences by hand
        x_test_defense = x_test
        x_test_defense, _ = fs(x_test_defense, y_test)
        x_test_defense, _ = jpeg(x_test_defense, y_test)
        x_test_defense, _ = smooth(x_test_defense, y_test)
        preds_check = self.model_mnist._model.predict(x_test_defense)

        # Check that the prediction results match
        self.assertTrue((preds_classifier - preds_check <= 1e-5).all())

    def test_class_gradient(self):
        (_, _), (x_test, _) = self.mnist
        classifier = self.model_mnist

        # Test all gradients label
        grads = classifier.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test 1 gradient label = 5
        grads = classifier.class_gradient(x_test, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = classifier.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_loss_gradient(self):
        (_, _), (x_test, y_test) = self.mnist
        classifier = self.model_mnist

        # Test gradient
        grads = classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_functional_model(self):
        self._test_functional_model(custom_activation=True)
        self._test_functional_model(custom_activation=False)

    def _test_functional_model(self, custom_activation=True):
        # Need to update the functional_model code to produce a model with more than one input and output layers...
        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=1, output_layer=1,
                                      custom_activation=custom_activation)
        self.assertTrue(keras_model._input.name, "input1")
        self.assertTrue(keras_model._output.name, "output1")
        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=0, output_layer=0,
                                      custom_activation=custom_activation)
        self.assertTrue(keras_model._input.name, "input0")
        self.assertTrue(keras_model._output.name, "output0")

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]

        classifier = self.model_mnist
        self.assertEqual(len(classifier.layer_names), 3)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            act_i = classifier.get_activations(x_test, i)
            act_name = classifier.get_activations(x_test, name)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)

    def test_resnet(self):
        import os

        from keras.applications.resnet50 import ResNet50, decode_predictions
        from keras.preprocessing.image import load_img, img_to_array

        keras.backend.set_learning_phase(0)
        model = ResNet50(weights='imagenet')
        classifier = KerasClassifier(model, clip_values=(0, 255))

        # Load image from file
        image = img_to_array(load_img(os.path.join(self.test_dir, 'test.jpg'), target_size=(224, 224)))
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        label = decode_predictions(classifier.predict(image))[0][0]
        self.assertEqual(label[1], 'Weimaraner')

    def test_learning_phase(self):
        classifier = self.model_mnist

        self.assertFalse(hasattr(classifier, '_learning_phase'))
        classifier.set_learning_phase(False)
        self.assertFalse(classifier.learning_phase)
        classifier.set_learning_phase(True)
        self.assertTrue(classifier.learning_phase)
        self.assertTrue(hasattr(classifier, '_learning_phase'))

    def test_save(self):
        import os

        path = 'tmp'
        filename = 'model.h5'
        self.model_mnist.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename)))

        # Remove saved file
        os.remove(os.path.join(path, filename))

    def test_pickle(self):
        import os
        filename = 'my_classifier.p'
        from art import DATA_PATH
        full_path = os.path.join(DATA_PATH, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        import pickle
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
        repr_ = repr(self.model_mnist)
        self.assertIn('art.classifiers.keras.KerasClassifier', repr_)
        self.assertIn('use_logits=False, channel_index=3', repr_)
        self.assertIn('clip_values=(0, 1), defences=None, preprocessing=(0, 1)', repr_)
        self.assertIn('input_layer=0, output_layer=0', repr_)
