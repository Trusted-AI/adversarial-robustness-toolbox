# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
import os
import pickle
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from art.config import ART_DATA_PATH
from art.data_generators import KerasDataGenerator
from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.estimators.classification.keras import KerasClassifier, generator_fit
from art.utils import Deprecated
from tests.utils import TestBase, get_image_classifier_kr_tf, get_image_classifier_kr_tf_with_wildcard, master_seed

if tf.__version__[0] == "2":
    tf.compat.v1.disable_eager_execution()

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

    def test_loss_gradient_with_wildcard(self):
        classifier = get_image_classifier_kr_tf_with_wildcard()

        # Test gradient
        shapes = [(1, 10, 1), (1, 20, 1)]
        for shape in shapes:
            x = np.random.normal(size=shape)
            loss_gradient = classifier.loss_gradient(x, y=[1])
            self.assertEqual(loss_gradient.shape, shape)

            class_gradient = classifier.class_gradient(x, 0)
            self.assertEqual(class_gradient[0].shape, shape)

    def test_functional_model(self):
        # Need to update the functional_model code to produce a model with more than one input and output layers...
        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=1, output_layer=1)
        self.assertTrue(keras_model._input.name, "input1")
        self.assertTrue(keras_model._output.name, "output1")

        keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=0, output_layer=0)
        self.assertTrue(keras_model._input.name, "input0")
        self.assertTrue(keras_model._output.name, "output0")

    def test_learning_phase(self):
        classifier = get_image_classifier_kr_tf()

        self.assertFalse(hasattr(classifier, "_learning_phase"))
        classifier.set_learning_phase(False)
        self.assertFalse(classifier.learning_phase)
        classifier.set_learning_phase(True)
        self.assertTrue(classifier.learning_phase)
        self.assertTrue(hasattr(classifier, "_learning_phase"))

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
        self.assertEqual(keras_model._channels_first, loaded._channels_first)
        self.assertEqual(keras_model._use_logits, loaded._use_logits)
        self.assertEqual(keras_model._input_layer, loaded._input_layer)
        self.assertEqual(self.functional_model.get_config(), loaded._model.get_config())
        self.assertTrue(isinstance(loaded.preprocessing_defences[0], FeatureSqueezing))

        os.remove(full_path)

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
