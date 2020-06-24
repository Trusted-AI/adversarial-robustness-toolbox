import keras
import logging
import numpy as np
from os import listdir, path
import tempfile
import warnings

from art.utils import Deprecated

from tests.utils import ExpectedValue

logger = logging.getLogger(__name__)


def is_keras_2_3():
    if int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3:
        return True
    return False


def test_fit(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    labels = np.argmax(y_test_mnist, axis=1)
    classifier, sess = get_image_classifier_list(one_classifier=True)
    if classifier is not None:
        accuracy = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
        np.testing.assert_array_almost_equal(accuracy, 0.32, decimal=0.06)

        classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=2)
        accuracy_2 = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
        np.testing.assert_array_almost_equal(accuracy_2, 0.73, decimal=0.06)


def test_shapes(get_default_mnist_subset, get_image_classifier_list):
    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, sess = get_image_classifier_list(one_classifier=True)

    if classifier is not None:
        predictions = classifier.predict(x_test_mnist)
        assert predictions.shape == y_test_mnist.shape

        assert classifier.nb_classes == 10

        class_gradients = classifier.class_gradient(x_test_mnist[:11])
        assert class_gradients.shape == tuple([11, 10] + list(x_test_mnist[1].shape))

        loss_gradients = classifier.loss_gradient(x_test_mnist[:11], y_test_mnist[:11])
        assert loss_gradients.shape == x_test_mnist[:11].shape


def test_fit_image_generator(framework, is_tf_version_2, get_image_classifier_list, image_data_generator,
                             get_default_mnist_subset):
    if framework == "tensorflow" and is_tf_version_2:
        return

    classifier, sess = get_image_classifier_list(one_classifier=True)

    if classifier is not None:
        data_gen = image_data_generator(sess=sess)

        (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        true_class = np.argmax(y_test_mnist, axis=1)

        predictions = classifier.predict(x_test_mnist)
        prediction_class = np.argmax(predictions, axis=1)
        pre_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
        logger.info("Accuracy: %.2f%%", (pre_fit_accuracy * 100))

        np.testing.assert_array_almost_equal(pre_fit_accuracy, 0.32, decimal=0.06, )

        classifier.fit_generator(generator=data_gen, nb_epochs=2)
        predictions = classifier.predict(x_test_mnist)
        prediction_class = np.argmax(predictions, axis=1)
        post_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
        logger.info("Accuracy after fitting classifier with generator: %.2f%%", (post_fit_accuracy * 100))

        np.testing.assert_array_almost_equal(post_fit_accuracy, 0.68, decimal=0.06, )


def test_loss_gradient(framework, is_tf_version_2, get_default_mnist_subset, get_image_classifier_list,
                       expected_values, mnist_shape):
    if framework == "keras" and is_keras_2_3() is False:
        # Keras 2.2 does not support creating classifiers with logits=True so skipping this test
        return

    (expected_gradients_1, expected_gradients_2) = expected_values
    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    if classifier is not None:
        gradients = classifier.loss_gradient(x_test_mnist, y_test_mnist)

        assert gradients.shape == (x_test_mnist.shape[0],) + mnist_shape

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, :, 14]
        else:
            sub_gradients = gradients[0, :, 14, 0]

        np.testing.assert_array_almost_equal(sub_gradients, expected_gradients_1[0], decimal=expected_gradients_1[1], )

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 14, :]
        else:
            sub_gradients = gradients[0, 14, :, 0]

        np.testing.assert_array_almost_equal(sub_gradients, expected_gradients_2[0], decimal=expected_gradients_2[1], )


def test_layers(get_default_mnist_subset, framework, is_tf_version_2, get_image_classifier_list):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True)
        if classifier is not None:
            (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

            if framework == "tensorflow" and is_tf_version_2:
                raise NotImplementedError(
                    "fw_agnostic_backend_test_layers not implemented for framework {0}".format(framework))

            layer_count = 3
            if framework == "pytorch":
                layer_count = 1
            if framework == "tensorflow":
                layer_count = 5

            if layer_count is not None:
                assert len(classifier.layer_names) == layer_count

            batch_size = 128
            for i, name in enumerate(classifier.layer_names):
                activation_i = classifier.get_activations(x_test_mnist, i, batch_size=batch_size)
                activation_name = classifier.get_activations(x_test_mnist, name, batch_size=batch_size)
                np.testing.assert_array_equal(activation_name, activation_i)
    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


def test_predict(request, framework, get_default_mnist_subset, get_image_classifier_list,
                 expected_values):
    if framework == "keras" and is_keras_2_3() is False:
        # Keras 2.2 does not support creating classifiers with logits=True so skipping this test
        return

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    if classifier is not None:
        y_predicted = classifier.predict(x_test_mnist[0:1])

        np.testing.assert_array_almost_equal(y_predicted, expected_values, decimal=4)


def test_nb_classes(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    if classifier is not None:
        assert classifier.nb_classes == 10


def test_input_shape(get_image_classifier_list, mnist_shape):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    if classifier is not None:
        assert classifier.input_shape == mnist_shape


def test_save(get_image_classifier_list):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True)
        if classifier is not None:
            t_file = tempfile.NamedTemporaryFile()
            model_path = t_file.name
            t_file.close()
            filename = "model_to_save"
            classifier.save(filename, path=model_path)

            assert path.exists(model_path)

            created_model = False

            for file in listdir(model_path):
                if filename in file:
                    created_model = True
            assert created_model

    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


def test_repr(framework, is_tf_version_2, get_image_classifier_list):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True)
        if classifier is not None:
            message_list = None
            if framework == "pytorch":
                message_list = [
                    "art.estimators.classification.pytorch.PyTorchClassifier",
                    "(conv): Conv2d(1, 1, kernel_size=(7, 7), stride=(1, 1))",
                    "(pool): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)",
                    "(fullyconnected): Linear(in_features=25, out_features=10, bias=True)",
                    "loss=CrossEntropyLoss(), optimizer=Adam",
                    "input_shape=(1, 28, 28), nb_classes=10, channel_index",
                    "clip_values=array([0., 1.], dtype=float32",
                    "preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1)"
                ]

            elif framework == "keras":
                message_list = [
                    "art.estimators.classification.keras.KerasClassifier",
                    f"use_logits=False, channel_index={Deprecated}, channels_first=False",
                    "clip_values=array([0., 1.], dtype=float32), preprocessing_defences=None, "
                    "postprocessing_defences=None, "
                    "preprocessing=(0, 1)",
                    "input_layer=0, output_layer=0",
                ]

            elif framework == "tensorflow":
                if is_tf_version_2:
                    message_list = [
                        "TensorFlowV2Classifier",
                        "model=",
                        "nb_classes=10",
                        "input_shape=(28, 28, 1)",
                        "loss_object=<tensorflow.python.keras.losses." "SparseCategoricalCrossentropy",
                        "train_step=<function get_image_classifier_tf_v2." "<locals>.train_step",
                        f"channel_index={Deprecated}, channels_first=False, ",
                        "clip_values=array([0., 1.], dtype=float32), ",
                        "preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1))",
                    ]

                else:

                    message_list = [
                        "TensorFlowClassifier",
                        "input_ph=<tf.Tensor 'Placeholder:0' shape=(?, 28, 28, 1) dtype=float32>",
                        "output=<tf.Tensor 'Softmax:0' shape=(?, 10) dtype=float32>,",
                        "labels_ph=<tf.Tensor 'Placeholder_1:0' shape=(?, 10) dtype=float32>",
                        "train=<tf.Operation 'Adam' type=NoOp>",
                        "loss=<tf.Tensor 'Mean:0' shape=() dtype=float32>",
                        "learning=None",
                        "sess=<tensorflow.python.client.session.Session object",
                        "TensorFlowClassifier",
                        f"channel_index={Deprecated}, channels_first=False, ",
                        "clip_values=array([0., 1.], dtype=float32), ",
                        "preprocessing_defences=None, postprocessing_defences=None, ",
                        "preprocessing=(0, 1))",
                    ]

            if message_list is None:
                raise NotImplementedError(
                    "fw_agnostic_backend_test_repr not implemented for framework {0}".format(framework))

            repr_ = repr(classifier)
            for message in message_list:
                assert message in repr_, "{0}: was not contained within repr".format(message)

    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


def test_class_gradient(framework, get_image_classifier_list, get_default_mnist_subset, expected_values, mnist_shape):
    if framework == "keras" and is_keras_2_3() is False:
        # Keras 2.2 does not support creating classifiers with logits=True so skipping this test
        return

    (grad_1_all_labels, grad_2_all_labels, grad_1_label5, grad_2_label5, grad_1_labelArray, grad_2_labelArray,
     labels) = expected_values

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
    if classifier is not None:
        # Test all gradients label
        gradients = classifier.class_gradient(x_test_mnist)

        new_shape = (x_test_mnist.shape[0], 10,) + mnist_shape
        if framework == "pytorch":

            assert gradients.shape == new_shape
        else:
            assert gradients.shape == new_shape

        if framework == "pytorch":
            sub_gradients = gradients[0, 5, 0, 14, :]  # expected_gradients_1_all_labels
        else:
            sub_gradients = gradients[0, 5, 14, :, 0]

        np.testing.assert_array_almost_equal(sub_gradients, grad_1_all_labels[0], decimal=grad_1_all_labels[1], )

        if framework == "pytorch":
            sub_gradients = gradients[0, 5, 0, :, 14]  # expected_gradients_2_all_labels
        else:
            sub_gradients = gradients[0, 5, :, 14, 0]

        np.testing.assert_array_almost_equal(sub_gradients, grad_2_all_labels[0], decimal=grad_2_all_labels[1], )

        # Test 1 gradient label = 5
        gradients = classifier.class_gradient(x_test_mnist, label=5)

        new_shape = (x_test_mnist.shape[0], 1,) + mnist_shape
        if framework == "pytorch":
            assert gradients.shape == new_shape
        else:
            assert gradients.shape == new_shape

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, 14, :]  # expected_gradients_1_label5
        else:
            sub_gradients = gradients[0, 0, 14, :, 0]

        np.testing.assert_array_almost_equal(sub_gradients, grad_1_label5[0], decimal=grad_1_label5[1], )

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, :, 14]  # expected_gradients_2_all_labels
        else:
            sub_gradients = gradients[0, 0, :, 14, 0]

        np.testing.assert_array_almost_equal(sub_gradients, grad_2_label5[0], decimal=grad_2_label5[1], )

        # # Test a set of gradients label = array
        # # label = np.random.randint(5, size=self.n_test)
        gradients = classifier.class_gradient(x_test_mnist, label=labels)

        new_shape = (x_test_mnist.shape[0], 1,) + mnist_shape
        if framework == "pytorch":
            assert gradients.shape == new_shape
        else:
            assert gradients.shape == new_shape

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, 14, :]  # expected_gradients_1_labelArray
        else:
            sub_gradients = gradients[0, 0, 14, :, 0]

        np.testing.assert_array_almost_equal(sub_gradients, grad_1_labelArray[0], decimal=grad_1_labelArray[1], )

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, :, 14]  # expected_gradients_2_labelArray
        else:
            sub_gradients = gradients[0, 0, :, 14, 0]

        np.testing.assert_array_almost_equal(sub_gradients, grad_2_labelArray[0], decimal=grad_2_labelArray[1], )
