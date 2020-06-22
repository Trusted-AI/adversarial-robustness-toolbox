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
import os

import keras
import numpy as np
import pytest
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from art.data_generators import KerasDataGenerator
from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.estimators.classification.keras import KerasClassifier, generator_fit
from art.utils import Deprecated
from tests.classifiersFrameworks.utils import (
    backend_test_fit_generator
)
from tests.utils import ExpectedValue

logger = logging.getLogger(__name__)


# %TODO classifier = get_image_classifier_kr() needs to be a fixture I think maybe?


def _functional_model():
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
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
        loss_weights=[1.0, 1.0],
    )

    return model


# TODO this should be scope="module" no point doing it for each function
@pytest.fixture()
def get_functional_model(get_default_mnist_subset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    # Load small Keras model
    functional_model = _functional_model()
    functional_model.fit([x_train_mnist, x_train_mnist], [y_train_mnist, y_train_mnist], nb_epoch=3)

    yield functional_model


@pytest.mark.only_with_platform("keras")
def test_fit(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    labels = np.argmax(y_test_mnist, axis=1)
    classifier, _ = get_image_classifier_list(one_classifier=True)
    accuracy = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
    logger.info("Accuracy: %.2f%%", (accuracy * 100))

    classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=2)
    accuracy_2 = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
    logger.info("Accuracy: %.2f%%", (accuracy_2 * 100))

    assert accuracy == 0.32
    np.testing.assert_array_almost_equal(accuracy_2, 0.73, decimal=0.06)


@pytest.mark.only_with_platform("keras")
def test_fit_generator(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    gen = generator_fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size)
    data_gen = KerasDataGenerator(iterator=gen, size=x_train_mnist.shape[0], batch_size=default_batch_size)

    classifier, _ = get_image_classifier_list(one_classifier=True)

    expected_values = {"pre_fit_accuracy": ExpectedValue(0.32, 0.06), "post_fit_accuracy": ExpectedValue(0.36, 0.06)}

    backend_test_fit_generator(expected_values, classifier, data_gen, get_default_mnist_subset, nb_epochs=3)


@pytest.mark.only_with_platform("keras")
def test_fit_image_generator(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True)
    labels_test = np.argmax(y_test_mnist, axis=1)
    accuracy = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels_test) / x_test_mnist.shape[0]
    logger.info("Accuracy: %.2f%%", (accuracy * 100))

    keras_gen = ImageDataGenerator(
        width_shift_range=0.075,
        height_shift_range=0.075,
        rotation_range=12,
        shear_range=0.075,
        zoom_range=0.05,
        fill_mode="constant",
        cval=0,
    )
    keras_gen.fit(x_train_mnist)
    data_gen = KerasDataGenerator(
        iterator=keras_gen.flow(x_train_mnist, y_train_mnist, batch_size=default_batch_size),
        size=x_train_mnist.shape[0],
        batch_size=default_batch_size,
    )
    classifier.fit_generator(generator=data_gen, nb_epochs=5)
    accuracy_2 = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels_test) / x_test_mnist.shape[0]
    logger.info("Accuracy: %.2f%%", (accuracy_2 * 100))

    assert accuracy == 0.32
    np.testing.assert_array_almost_equal(accuracy_2, 0.35, decimal=0.06)


@pytest.mark.only_with_platform("keras")
def test_fit_kwargs(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    def get_lr(_):
        return 0.01

    # Test a valid callback
    classifier, _ = get_image_classifier_list(one_classifier=True)
    kwargs = {"callbacks": [LearningRateScheduler(get_lr)]}
    classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=1, **kwargs)

    # Test failure for invalid parameters
    kwargs = {"epochs": 1}
    with pytest.raises(TypeError) as exception:
        classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=1, **kwargs)

    assert "multiple values for keyword argument" in str(exception.value)


@pytest.mark.only_with_platform("keras")
def test_shapes(get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True)

    predictions = classifier.predict(x_test_mnist)
    assert predictions.shape == y_test_mnist.shape

    assert classifier.nb_classes == 10

    class_gradients = classifier.class_gradient(x_test_mnist[:11])
    assert class_gradients.shape == tuple([11, 10] + list(x_test_mnist[1].shape))

    loss_gradients = classifier.loss_gradient(x_test_mnist[:11], y_test_mnist[:11])
    assert loss_gradients.shape == x_test_mnist[:11].shape


@pytest.mark.only_with_platform("keras")
def test_defences_predict(get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    clip_values = (0, 1)
    fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
    jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
    smooth = SpatialSmoothing()
    classifier_, _ = get_image_classifier_list(one_classifier=True)
    classifier = KerasClassifier(
        clip_values=clip_values, model=classifier_.model, preprocessing_defences=[fs, jpeg, smooth]
    )
    assert len(classifier.preprocessing_defences) == 3

    predictions_classifier = classifier.predict(x_test_mnist)

    # Apply the same defences by hand
    x_test_defense = x_test_mnist
    x_test_defense, _ = fs(x_test_defense, y_test_mnist)
    x_test_defense, _ = jpeg(x_test_defense, y_test_mnist)
    x_test_defense, _ = smooth(x_test_defense, y_test_mnist)
    classifier, _ = get_image_classifier_list(one_classifier=True)

    predictions_check = classifier.model.predict(x_test_defense)

    # Check that the prediction results match
    np.testing.assert_array_almost_equal(predictions_classifier, predictions_check, decimal=4)


@pytest.mark.only_with_platform("keras")
def test_functional_model(get_functional_model):
    functional_model = get_functional_model
    keras_model = KerasClassifier(functional_model, clip_values=(0, 1), input_layer=1, output_layer=1)
    assert keras_model._input.name == "input1:0"
    assert keras_model._output.name == "output1/Softmax:0"

    keras_model = KerasClassifier(functional_model, clip_values=(0, 1), input_layer=0, output_layer=0)
    assert keras_model._input.name == "input0:0"
    assert keras_model._output.name == "output0/Softmax:0"


@pytest.mark.only_with_platform("keras")
def test_resnet(create_test_image):
    image_file_path = create_test_image
    keras.backend.set_learning_phase(0)
    model = ResNet50(weights="imagenet")
    classifier = KerasClassifier(model, clip_values=(0, 255))

    image = img_to_array(load_img(image_file_path, target_size=(224, 224)))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    prediction = classifier.predict(image)
    label = decode_predictions(prediction)[0][0]

    assert label[1] == "Weimaraner"
    np.testing.assert_array_almost_equal(prediction[0, 178], 0.2658045, decimal=3)


@pytest.mark.only_with_platform("keras")
def test_learning_phase(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    assert hasattr(classifier, "_learning_phase") is False
    classifier.set_learning_phase(False)
    assert classifier.learning_phase is False
    classifier.set_learning_phase(True)
    assert classifier.learning_phase
    assert hasattr(classifier, "_learning_phase")


# def test_pickle(self):
#     filename = 'my_classifier.p'
#     full_path = os.path.join(ART_DATA_PATH, filename)
#     folder = os.path.split(full_path)[0]
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#
#     fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
#     keras_model = KerasClassifier(self.functional_model, clip_values=(0, 1), input_layer=1, output_layer=1,
#                                   defences=fs)
#     with open(full_path, 'wb') as save_file:
#         pickle.dump(keras_model, save_file)
#
#     # Unpickle:
#     with open(full_path, 'rb') as load_file:
#         loaded = pickle.load(load_file)
#
#     self.assertEqual(keras_model._clip_values, loaded._clip_values)
#     self.assertEqual(keras_model._channel_index, loaded._channel_index)
#     self.assertEqual(keras_model._use_logits, loaded._use_logits)
#     self.assertEqual(keras_model._input_layer, loaded._input_layer)
#     self.assertEqual(self.functional_model.get_config(), loaded._model.get_config())
#     self.assertTrue(isinstance(loaded.defences[0], FeatureSqueezing))
#
#     os.remove(full_path)


@pytest.mark.only_with_platform("keras")
def test_loss_functions(get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

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

        classifier, _ = get_image_classifier_list(
            one_classifier=True, loss_name=_loss_name, loss_type=_loss_type, from_logits=_from_logits
        )

        y_test_pred = np.argmax(classifier.predict(x=x_test_mnist), axis=1)
        np.testing.assert_array_equal(y_test_pred, _y_test_pred_expected)

        class_gradient = classifier.class_gradient(x_test_mnist, label=5)
        np.testing.assert_array_almost_equal(class_gradient[99, 0, 14, :, 0], _class_gradient_probabilities_expected)

        loss_gradient = classifier.loss_gradient(x=x_test_mnist, y=y_test_mnist)
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

    for loss_type in ["function_losses"]:
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

    for loss_type in ["label", "function_losses", "function_backend"]:
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

    for loss_type in ["function_backend"]:
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

    for loss_type in ["label", "function_losses", "function_backend"]:
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

    for loss_type in ["function_backend"]:
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

    for loss_type in ["function_losses"]:
        logger.info("loss_name: {}, loss_type: {}, output: logits".format(loss_name, loss_type))

        _run_tests(
            loss_name,
            loss_type,
            y_test_pred_expected,
            class_gradient_probabilities_expected,
            loss_gradient_expected,
            _from_logits=False,
        )


if __name__ == "__main__":
    pytest.cmdline.main("-q {} --mlFramework=keras --durations=0".format(__file__).split(" "))
