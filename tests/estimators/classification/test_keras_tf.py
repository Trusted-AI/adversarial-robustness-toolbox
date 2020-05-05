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
import pickle

import tensorflow as tf

if tf.__version__[0] == "2":
    tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

import numpy as np

from art.config import ART_DATA_PATH
from art.estimators.classification.keras import KerasClassifier, generator_fit
from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.data_generators import KerasDataGenerator

from tests.utils import TestBase, master_seed, get_image_classifier_kr_tf

logger = logging.getLogger(__name__)


class TestKerasClassifierTensorFlow(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        # Load small Keras model
        cls.functional_model = cls.functional_model()
        cls.functional_model.fit(
            [cls.x_train_mnist, cls.x_train_mnist], [cls.y_train_mnist, cls.y_train_mnist], epochs=3
        )

    def setUp(self):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUp()

    @staticmethod
    def functional_model():
        in_layer = Input(shape=(28, 28, 1), name="input0")
        layer = Conv2D(32, kernel_size=(3, 3), activation="relu")(in_layer)
        layer = Conv2D(64, (3, 3), activation="relu")(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)
        layer = Dropout(0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation="relu")(layer)
        layer = Dropout(0.5)(layer)
        out_layer = Dense(10, activation="softmax", name="output0")(layer)

        in_layer_2 = Input(shape=(28, 28, 1), name="input1")
        layer = Conv2D(32, kernel_size=(3, 3), activation="relu")(in_layer_2)
        layer = Conv2D(64, (3, 3), activation="relu")(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)
        layer = Dropout(0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation="relu")(layer)
        layer = Dropout(0.5)(layer)
        out_layer_2 = Dense(10, activation="softmax", name="output1")(layer)

        model = Model(inputs=[in_layer, in_layer_2], outputs=[out_layer, out_layer_2])

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adadelta(),
            metrics=["accuracy"],
            loss_weights=[1.0, 1.0],
        )

        return model

    def test_fit(self):
        labels = np.argmax(self.y_test_mnist, axis=1)
        classifier = get_image_classifier_kr_tf()
        acc = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / self.n_test
        logger.info("Accuracy: %.2f%%", (acc * 100))

        classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=self.batch_size, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / self.n_test
        logger.info("Accuracy: %.2f%%", (acc2 * 100))

        self.assertEqual(acc, 0.32)
        self.assertEqual(acc2, 0.74)

    def test_fit_generator(self):
        labels = np.argmax(self.y_test_mnist, axis=1)
        classifier = get_image_classifier_kr_tf()
        acc = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / self.n_test
        logger.info("Accuracy: %.2f%%", (acc * 100))

        gen = generator_fit(self.x_train_mnist, self.y_train_mnist, batch_size=self.batch_size)
        data_gen = KerasDataGenerator(iterator=gen, size=self.n_train, batch_size=self.batch_size)
        classifier.fit_generator(generator=data_gen, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / self.n_test
        logger.info("Accuracy: %.2f%%", (acc2 * 100))

        self.assertEqual(acc, 0.32)
        self.assertAlmostEqual(acc2, 0.70, delta=0.15)

    def test_fit_image_generator(self):
        master_seed(seed=1234)
        labels_test = np.argmax(self.y_test_mnist, axis=1)
        classifier = get_image_classifier_kr_tf()
        acc = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels_test) / self.n_test
        logger.info("Accuracy: %.2f%%", (acc * 100))

        keras_gen = ImageDataGenerator(
            width_shift_range=0.075,
            height_shift_range=0.075,
            rotation_range=12,
            shear_range=0.075,
            zoom_range=0.05,
            fill_mode="constant",
            cval=0,
        )
        keras_gen.fit(self.x_train_mnist)
        data_gen = KerasDataGenerator(
            iterator=keras_gen.flow(self.x_train_mnist, self.y_train_mnist, batch_size=self.batch_size),
            size=self.n_train,
            batch_size=self.batch_size,
        )
        classifier.fit_generator(generator=data_gen, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels_test) / self.n_test
        logger.info("Accuracy: %.2f%%", (acc2 * 100))

        self.assertEqual(acc, 0.32)
        self.assertAlmostEqual(acc2, 0.69, delta=0.02)

    def test_fit_kwargs(self):
        def get_lr(_):
            return 0.01

        # Test a valid callback
        classifier = get_image_classifier_kr_tf()
        kwargs = {"callbacks": [LearningRateScheduler(get_lr)]}
        classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=self.batch_size, nb_epochs=1, **kwargs)

        # Test failure for invalid parameters
        kwargs = {"epochs": 1}
        with self.assertRaises(TypeError) as context:
            classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=self.batch_size, nb_epochs=1, **kwargs)

        self.assertIn("multiple values for keyword argument", str(context.exception))

    def test_shapes(self):
        classifier = get_image_classifier_kr_tf()

        predictions = classifier.predict(self.x_test_mnist)
        self.assertEqual(predictions.shape, self.y_test_mnist.shape)

        self.assertEqual(classifier.nb_classes, 10)

        class_gradients = classifier.class_gradient(self.x_test_mnist[:11])
        self.assertEqual(class_gradients.shape, tuple([11, 10] + list(self.x_test_mnist[1].shape)))

        loss_gradients = classifier.loss_gradient(self.x_test_mnist[:11], self.y_test_mnist[:11])
        self.assertEqual(loss_gradients.shape, self.x_test_mnist[:11].shape)

    def test_defences_predict(self):
        clip_values = (0, 1)
        fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
        jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
        smooth = SpatialSmoothing()

        classifier_ = get_image_classifier_kr_tf()
        classifier = KerasClassifier(
            clip_values=clip_values, model=classifier_._model, preprocessing_defences=[fs, jpeg, smooth]
        )
        self.assertEqual(len(classifier.preprocessing_defences), 3)

        predictions_classifier = classifier.predict(self.x_test_mnist)

        # Apply the same defences by hand
        x_test_defense = self.x_test_mnist
        x_test_defense, _ = fs(x_test_defense, self.y_test_mnist)
        x_test_defense, _ = jpeg(x_test_defense, self.y_test_mnist)
        x_test_defense, _ = smooth(x_test_defense, self.y_test_mnist)
        classifier = get_image_classifier_kr_tf()
        predictions_check = classifier._model.predict(x_test_defense)

        # Check that the prediction results match
        np.testing.assert_array_almost_equal(predictions_classifier, predictions_check, decimal=4)

    def test_class_gradient(self):
        classifier = get_image_classifier_kr_tf()

        # Test all gradients label
        gradients = classifier.class_gradient(self.x_test_mnist)

        self.assertTrue(gradients.shape == (self.n_test, 10, 28, 28, 1))

        expected_gradients_1 = np.asarray(
            [
                -1.0557447e-03,
                -1.0079544e-03,
                -7.7426434e-04,
                1.7387432e-03,
                2.1773507e-03,
                5.0880699e-05,
                1.6497371e-03,
                2.6113100e-03,
                6.0904310e-03,
                4.1080985e-04,
                2.5268078e-03,
                -3.6661502e-04,
                -3.0568996e-03,
                -1.1665225e-03,
                3.8904310e-03,
                3.1726385e-04,
                1.3203260e-03,
                -1.1720930e-04,
                -1.4315104e-03,
                -4.7676818e-04,
                9.7251288e-04,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 5, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray(
            [
                -0.00367321,
                -0.0002892,
                0.00037825,
                -0.00053344,
                0.00192121,
                0.00112047,
                0.0023135,
                0.0,
                0.0,
                -0.00391743,
                -0.0002264,
                0.00238103,
                -0.00073711,
                0.00270405,
                0.00389043,
                0.00440818,
                -0.00412769,
                -0.00441794,
                0.00081916,
                -0.00091284,
                0.00119645,
                -0.00849089,
                0.00547925,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 5, :, 14, 0], expected_gradients_2, decimal=4)

        # Test 1 gradient label = 5
        gradients = classifier.class_gradient(self.x_test_mnist, label=5)

        self.assertTrue(gradients.shape == (self.n_test, 1, 28, 28, 1))

        expected_gradients_1 = np.asarray(
            [
                -1.0557447e-03,
                -1.0079544e-03,
                -7.7426434e-04,
                1.7387432e-03,
                2.1773507e-03,
                5.0880699e-05,
                1.6497371e-03,
                2.6113100e-03,
                6.0904310e-03,
                4.1080985e-04,
                2.5268078e-03,
                -3.6661502e-04,
                -3.0568996e-03,
                -1.1665225e-03,
                3.8904310e-03,
                3.1726385e-04,
                1.3203260e-03,
                -1.1720930e-04,
                -1.4315104e-03,
                -4.7676818e-04,
                9.7251288e-04,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray(
            [
                -0.00367321,
                -0.0002892,
                0.00037825,
                -0.00053344,
                0.00192121,
                0.00112047,
                0.0023135,
                0.0,
                0.0,
                -0.00391743,
                -0.0002264,
                0.00238103,
                -0.00073711,
                0.00270405,
                0.00389043,
                0.00440818,
                -0.00412769,
                -0.00441794,
                0.00081916,
                -0.00091284,
                0.00119645,
                -0.00849089,
                0.00547925,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=self.n_test)
        gradients = classifier.class_gradient(self.x_test_mnist, label=label)

        self.assertTrue(gradients.shape == (self.n_test, 1, 28, 28, 1))

        expected_gradients_1 = np.asarray(
            [
                5.0867125e-03,
                4.8564528e-03,
                6.1040390e-03,
                8.6531248e-03,
                -6.0958797e-03,
                -1.4114540e-02,
                -7.1085989e-04,
                -5.0330814e-04,
                1.2943064e-02,
                8.2416134e-03,
                -1.9859476e-04,
                -9.8109958e-05,
                -3.8902222e-03,
                -1.2945873e-03,
                7.5137997e-03,
                1.7720886e-03,
                3.1399424e-04,
                2.3657181e-04,
                -3.0891625e-03,
                -1.0211229e-03,
                2.0828887e-03,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray(
            [
                -0.00195835,
                -0.00134457,
                -0.00307221,
                -0.00340564,
                0.00175022,
                -0.00239714,
                -0.00122619,
                0.0,
                0.0,
                -0.00520899,
                -0.00046105,
                0.00414874,
                -0.00171095,
                0.00429184,
                0.0075138,
                0.00792442,
                0.0019566,
                0.00035517,
                0.00504575,
                -0.00037397,
                0.00022343,
                -0.00530034,
                0.0020528,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)

    def test_loss_gradient(self):
        classifier = get_image_classifier_kr_tf()

        # Test gradient
        gradients = classifier.loss_gradient(self.x_test_mnist, self.y_test_mnist)

        self.assertTrue(gradients.shape == (self.n_test, 28, 28, 1))

        expected_gradients_1 = np.asarray(
            [
                0.0559206,
                0.05338925,
                0.0648919,
                0.07925165,
                -0.04029291,
                -0.11281465,
                0.01850601,
                0.00325054,
                0.08163195,
                0.03333949,
                0.031766,
                -0.02420463,
                -0.07815556,
                -0.04698735,
                0.10711591,
                0.04086434,
                -0.03441073,
                0.01071284,
                -0.04229195,
                -0.01386157,
                0.02827487,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray(
            [
                0.00210803,
                0.00213919,
                0.00520981,
                0.00548001,
                -0.0023059,
                0.00432077,
                0.00274945,
                0.0,
                0.0,
                -0.0583441,
                -0.00616604,
                0.0526219,
                -0.02373985,
                0.05273106,
                0.10711591,
                0.12773865,
                0.0689289,
                0.01337799,
                0.10032021,
                0.01681096,
                -0.00028647,
                -0.05588859,
                0.01474165,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, :, 14, 0], expected_gradients_2, decimal=4)

    def test_functional_model(self):
        # Need to update the functional_model code to produce a model with more than one input and output layers...
        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=1, output_layer=1)
        self.assertTrue(keras_model._input.name, "input1")
        self.assertTrue(keras_model._output.name, "output1")

        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=0, output_layer=0)
        self.assertTrue(keras_model._input.name, "input0")
        self.assertTrue(keras_model._output.name, "output0")

    def test_layers(self):
        classifier = get_image_classifier_kr_tf()
        self.assertEqual(len(classifier.layer_names), 3)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            act_i = classifier.get_activations(self.x_test_mnist, i, batch_size=128)
            act_name = classifier.get_activations(self.x_test_mnist, name, batch_size=128)
            np.testing.assert_array_equal(act_name, act_i)

    def test_learning_phase(self):
        classifier = get_image_classifier_kr_tf()

        self.assertFalse(hasattr(classifier, "_learning_phase"))
        classifier.set_learning_phase(False)
        self.assertFalse(classifier.learning_phase)
        classifier.set_learning_phase(True)
        self.assertTrue(classifier.learning_phase)
        self.assertTrue(hasattr(classifier, "_learning_phase"))

    def test_save(self):
        path = "tmp"
        filename = "model.h5"
        classifier = get_image_classifier_kr_tf()
        classifier.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename)))

        # Remove saved file
        os.remove(os.path.join(path, filename))

    def test_pickle(self):
        filename = "my_classifier.p"
        full_path = os.path.join(ART_DATA_PATH, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
        keras_model = KerasClassifier(
            self.functional_model, clip_values=(0, 1), input_layer=1, output_layer=1, preprocessing_defences=fs
        )
        with open(full_path, "wb") as save_file:
            pickle.dump(keras_model, save_file)

        # Unpickle:
        with open(full_path, "rb") as load_file:
            loaded = pickle.load(load_file)

        np.testing.assert_equal(keras_model._clip_values, loaded._clip_values)
        self.assertEqual(keras_model._channel_index, loaded._channel_index)
        self.assertEqual(keras_model._use_logits, loaded._use_logits)
        self.assertEqual(keras_model._input_layer, loaded._input_layer)
        self.assertEqual(self.functional_model.get_config(), loaded._model.get_config())
        self.assertTrue(isinstance(loaded.preprocessing_defences[0], FeatureSqueezing))

        os.remove(full_path)

    def test_repr(self):
        classifier = get_image_classifier_kr_tf()
        repr_ = repr(classifier)
        self.assertIn("art.estimators.classification.keras.KerasClassifier", repr_)
        self.assertIn("use_logits=False, channel_index=3", repr_)
        self.assertIn(
            "clip_values=array([0., 1.], dtype=float32), preprocessing_defences=None, postprocessing_defences=None, "
            "preprocessing=(0, 1)",
            repr_,
        )
        self.assertIn("input_layer=0, output_layer=0", repr_)

    @unittest.skipIf(
        int(tf.__version__.split(".")[0]) == 1 and int(tf.__version__.split(".")[1]) < 14,
        reason="Only for TensorFlow 1.14 and higher.",
    )
    def test_loss_functions(self):

        # prediction and class_gradient should be independent of logits/probabilities and of loss function

        y_test_pred_expected = np.asarray(
            [
                7,
                1,
                1,
                4,
                4,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                1,
                4,
                4,
                7,
                8,
                4,
                4,
                4,
                4,
                3,
                4,
                4,
                7,
                4,
                4,
                4,
                7,
                4,
                3,
                4,
                7,
                0,
                7,
                7,
                1,
                1,
                7,
                7,
                4,
                0,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                7,
                4,
                4,
                4,
                4,
                4,
                1,
                4,
                4,
                7,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                3,
                0,
                7,
                4,
                0,
                1,
                7,
                4,
                4,
                7,
                4,
                4,
                4,
                4,
                4,
                7,
                4,
                4,
                4,
                4,
                4,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ]
        )

        class_gradient_probabilities_expected = np.asarray(
            [
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                2.3582461e-03,
                4.8802234e-04,
                1.6699843e-03,
                -6.4777887e-05,
                -1.4215634e-03,
                -1.3359448e-04,
                2.0448549e-03,
                2.8171093e-04,
                1.9665064e-04,
                1.5335126e-03,
                1.7000455e-03,
                -2.0136381e-04,
                6.4588618e-04,
                2.0524357e-03,
                2.1990810e-03,
                8.3692279e-04,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )

        class_gradient_logits_expected = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.08147776,
                0.01847786,
                0.07045883,
                -0.00269106,
                -0.03189164,
                0.01643312,
                0.1185048,
                0.02166386,
                0.00905327,
                0.06592228,
                0.04471018,
                -0.02879605,
                0.04668707,
                0.06856851,
                0.06857751,
                0.00657996,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        def _run_tests(
            _loss_name,
            _loss_type,
            _y_test_pred_expected,
            _class_gradient_probabilities_expected,
            _loss_gradient_expected,
            _from_logits,
        ):

            master_seed(1234)
            classifier = get_image_classifier_kr_tf(
                loss_name=_loss_name, loss_type=_loss_type, from_logits=_from_logits
            )

            y_test_pred = np.argmax(classifier.predict(x=self.x_test_mnist), axis=1)
            np.testing.assert_array_equal(y_test_pred, _y_test_pred_expected)

            class_gradient = classifier.class_gradient(self.x_test_mnist, label=5)
            np.testing.assert_array_almost_equal(
                class_gradient[99, 0, 14, :, 0], _class_gradient_probabilities_expected
            )

            loss_gradient = classifier.loss_gradient(x=self.x_test_mnist, y=self.y_test_mnist)
            np.testing.assert_array_almost_equal(loss_gradient[99, 14, :, 0], _loss_gradient_expected)

        # ================= #
        # categorical_hinge #
        # ================= #

        loss_name = "categorical_hinge"

        # loss_gradient should be the same for probabilities and logits but dependent on loss function

        loss_gradient_expected = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                -0.00644846,
                -0.00274792,
                -0.01334668,
                -0.01417109,
                0.03608133,
                0.00670766,
                0.02741555,
                -0.00938758,
                0.002172,
                0.01400783,
                -0.00224488,
                -0.01445661,
                0.01775403,
                0.00643058,
                0.00382644,
                -0.01077214,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # testing with probabilities

        for loss_type in ["function", "class"]:
            logger.info("loss_name: {}, loss_type: {}, output: probabilities".format(loss_name, loss_type))

            _run_tests(
                loss_name,
                loss_type,
                y_test_pred_expected,
                class_gradient_probabilities_expected,
                loss_gradient_expected,
                _from_logits=False,
            )

        # ======================== #
        # categorical_crossentropy #
        # ======================== #

        loss_name = "categorical_crossentropy"

        # loss_gradient should be the same for probabilities and logits but dependent on loss function

        loss_gradient_expected = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                -0.09573442,
                -0.0089094,
                0.01402334,
                0.0258659,
                0.08960329,
                0.10324767,
                0.10624839,
                0.06578761,
                -0.00018638,
                -0.01345262,
                -0.08770822,
                -0.04990875,
                0.04288402,
                -0.06845165,
                -0.08588978,
                -0.08277036,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # testing with probabilities

        for loss_type in ["label", "function", "class"]:
            logger.info("loss_name: {}, loss_type: {}, output: probabilities".format(loss_name, loss_type))

            _run_tests(
                loss_name,
                loss_type,
                y_test_pred_expected,
                class_gradient_probabilities_expected,
                loss_gradient_expected,
                _from_logits=False,
            )

        # testing with logits

        for loss_type in ["function", "class"]:
            logger.info("loss_name: {}, loss_type: {}, output: logits".format(loss_name, loss_type))

            _run_tests(
                loss_name,
                loss_type,
                y_test_pred_expected,
                class_gradient_logits_expected,
                loss_gradient_expected,
                _from_logits=True,
            )

        # =============================== #
        # sparse_categorical_crossentropy #
        # =============================== #

        loss_name = "sparse_categorical_crossentropy"

        # loss_gradient should be the same for probabilities and logits but dependent on loss function

        loss_gradient_expected = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                -0.09573442,
                -0.0089094,
                0.01402334,
                0.0258659,
                0.08960329,
                0.10324767,
                0.10624839,
                0.06578761,
                -0.00018638,
                -0.01345262,
                -0.08770822,
                -0.04990875,
                0.04288402,
                -0.06845165,
                -0.08588978,
                -0.08277036,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # testing with probabilities

        for loss_type in ["label", "function", "class"]:
            logger.info("loss_name: {}, loss_type: {}, output: probabilities".format(loss_name, loss_type))

            _run_tests(
                loss_name,
                loss_type,
                y_test_pred_expected,
                class_gradient_probabilities_expected,
                loss_gradient_expected,
                _from_logits=False,
            )

        # testing with logits

        for loss_type in ["function", "class"]:
            logger.info("loss_name: {}, loss_type: {}, output: logits".format(loss_name, loss_type))

            _run_tests(
                loss_name,
                loss_type,
                y_test_pred_expected,
                class_gradient_logits_expected,
                loss_gradient_expected,
                _from_logits=True,
            )

        # =========================== #
        # kullback_leibler_divergence #
        # =========================== #

        loss_name = "kullback_leibler_divergence"

        # loss_gradient should be the same for probabilities and logits but dependent on loss function

        loss_gradient_expected = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                -0.09573442,
                -0.0089094,
                0.01402334,
                0.0258659,
                0.08960329,
                0.10324767,
                0.10624839,
                0.06578761,
                -0.00018638,
                -0.01345262,
                -0.08770822,
                -0.04990875,
                0.04288402,
                -0.06845165,
                -0.08588978,
                -0.08277036,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # testing with probabilities

        for loss_type in ["function", "class"]:
            logger.info("loss_name: {}, loss_type: {}, output: probabilities".format(loss_name, loss_type))

            _run_tests(
                loss_name,
                loss_type,
                y_test_pred_expected,
                class_gradient_probabilities_expected,
                loss_gradient_expected,
                _from_logits=False,
            )


if __name__ == "__main__":
    unittest.main()
