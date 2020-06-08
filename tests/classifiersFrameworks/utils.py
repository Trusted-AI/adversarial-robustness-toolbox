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
import numpy as np

from art.utils import Deprecated

logger = logging.getLogger(__name__)


def backend_test_layers(
        get_mlFramework, get_default_mnist_subset, get_image_classifier_list, batch_size, layer_count=None
):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True)
    if layer_count is not None:
        assert len(classifier.layer_names) == layer_count

    for i, name in enumerate(classifier.layer_names):
        activation_i = classifier.get_activations(x_test_mnist, i, batch_size=batch_size)
        activation_name = classifier.get_activations(x_test_mnist, name, batch_size=batch_size)
        np.testing.assert_array_equal(activation_name, activation_i)


def fw_agnostic_backend_test_repr(framework, is_tf_version_2, classifier):

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
                f"channel_index={Deprecated}, channels_first=False, clip_values=array([0., 1.], dtype=float32), "
                "preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1))",
            ]
    else:
        message_list = [
            "TensorFlowClassifier",
            "input_ph=<tf.Tensor 'Placeholder:0' shape=(?, 28, 28, 1) dtype=float32>",
            "output=<tf.Tensor 'Softmax:0' shape=(?, 10) dtype=float32>",
            "labels_ph=<tf.Tensor 'Placeholder_1:0' shape=(?, 10) dtype=float32>",
            "train=<tf.Operation 'Adam' type=NoOp>",
            "loss=<tf.Tensor 'Mean:0' shape=() dtype=float32>",
            "learning=None",
            "sess=<tensorflow.python.client.session.Session object",
            "TensorFlowClassifier",
            f"channel_index={Deprecated}, channels_first=False, clip_values=array([0., 1.], dtype=float32), "
            "preprocessing_defences=None, postprocessing_defences=None, "
            "preprocessing=(0, 1))",
        ]

    if message_list is None:
        raise NotImplementedError("fw_agnostic_backend_test_repr not implemented for framework {0}".format(framework))

    repr_ = repr(classifier)
    for message in message_list:
        assert message in repr_, "{0}: was not contained within repr".format(message)


def fw_agnostic_backend_test_nb_classes(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    if classifier is not None:
        assert classifier.nb_classes == 10


def fw_agnostic_backend_test_input_shape(framework, get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    if classifier is not None:
        if framework == "pytorch":
            assert classifier.input_shape == (1, 28, 28)
        else:
            assert classifier.input_shape == (28, 28, 1)


def backend_test_class_gradient(framework, get_default_mnist_subset, classifier, expected_values, labels):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    # Test all gradients label
    gradients = classifier.class_gradient(x_test_mnist)

    if framework == "pytorch":
        assert gradients.shape == (x_test_mnist.shape[0], 10, 1, 28, 28)
    else:
        assert gradients.shape == (x_test_mnist.shape[0], 10, 28, 28, 1)

    if "expected_gradients_1_all_labels" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 5, 0, :, 14]
        else:
            sub_gradients = gradients[0, 5, 14, :, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1_all_labels"].value,
            decimal=expected_values["expected_gradients_1_all_labels"].decimals,
        )

    if "expected_gradients_2_all_labels" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 5, 0, 14, :]
        else:
            sub_gradients = gradients[0, 5, :, 14, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_2_all_labels"].value,
            decimal=expected_values["expected_gradients_2_all_labels"].decimals,
        )

    # Test 1 gradient label = 5
    gradients = classifier.class_gradient(x_test_mnist, label=5)

    if framework == "pytorch":
        assert gradients.shape == (x_test_mnist.shape[0], 1, 1, 28, 28)
    else:
        assert gradients.shape == (x_test_mnist.shape[0], 1, 28, 28, 1)

    if "expected_gradients_1_label5" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, :, 14]
        else:
            sub_gradients = gradients[0, 0, 14, :, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1_label5"].value,
            decimal=expected_values["expected_gradients_1_label5"].decimals,
        )

    if "expected_gradients_2_label5" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, 14, :]
        else:
            sub_gradients = gradients[0, 0, :, 14, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_2_label5"].value,
            decimal=expected_values["expected_gradients_2_label5"].decimals,
        )

    # # Test a set of gradients label = array
    # # label = np.random.randint(5, size=self.n_test)
    # gradients = classifier.class_gradient(x_test_mnist, label=labels)

    # backend_test_class_gradient_target(framework, classifier, get_default_mnist_subset, expected_values)
    if framework == "pytorch":
        assert gradients.shape == (x_test_mnist.shape[0], 1, 1, 28, 28)
    else:
        assert gradients.shape == (x_test_mnist.shape[0], 1, 28, 28, 1)

    if "expected_gradients_1_labelArray" in expected_values:

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, :, 14]
        else:
            sub_gradients = gradients[0, 0, 14, :, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1_labelArray"].value,
            decimal=expected_values["expected_gradients_1_labelArray"].decimals,
        )

    if "expected_gradients_2_labelArray" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, 14, :]
        else:
            sub_gradients = gradients[0, 0, :, 14, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_2_labelArray"].value,
            decimal=expected_values["expected_gradients_2_labelArray"].decimals,
        )


def backend_test_loss_gradient(framework, get_default_mnist_subset, get_image_classifier_list, expected_values):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True)

    # Test gradient
    gradients = classifier.loss_gradient(x_test_mnist, y_test_mnist)

    if framework == "pytorch":
        assert gradients.shape == (x_test_mnist.shape[0], 1, 28, 28)
    else:
        assert gradients.shape == (x_test_mnist.shape[0], 28, 28, 1)

    if framework == "pytorch":
        sub_gradients = gradients[0, 0, :, 14]
    else:
        sub_gradients = gradients[0, 14, :, 0]

    if "expected_gradients_1" in expected_values:
        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1"].value,
            decimal=expected_values["expected_gradients_1"].decimals,
        )

    if framework == "pytorch":
        sub_gradients = gradients[0, 0, 14, :]
    else:
        sub_gradients = gradients[0, :, 14, 0]

    if "expected_gradients_2" in expected_values:
        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_2"].value,
            decimal=expected_values["expected_gradients_2"].decimals,
        )


def backend_test_fit_generator(expected_values, classifier, data_gen, get_default_mnist_subset, nb_epochs):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    true_class = np.argmax(y_test_mnist, axis=1)

    predictions = classifier.predict(x_test_mnist)
    prediction_class = np.argmax(predictions, axis=1)
    pre_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
    logger.info("Accuracy: %.2f%%", (pre_fit_accuracy * 100))

    if "pre_fit_accuracy" in expected_values:
        np.testing.assert_array_almost_equal(
            pre_fit_accuracy,
            expected_values["pre_fit_accuracy"].value,
            decimal=expected_values["pre_fit_accuracy"].decimals,
        )

    classifier.fit_generator(generator=data_gen, nb_epochs=nb_epochs)
    predictions = classifier.predict(x_test_mnist)
    prediction_class = np.argmax(predictions, axis=1)
    post_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
    logger.info("Accuracy after fitting classifier with generator: %.2f%%", (post_fit_accuracy * 100))

    if "post_fit_accuracy" in expected_values:
        np.testing.assert_array_almost_equal(
            post_fit_accuracy,
            expected_values["post_fit_accuracy"].value,
            decimal=expected_values["post_fit_accuracy"].decimals,
        )
