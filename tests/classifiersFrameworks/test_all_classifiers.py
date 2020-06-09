import logging
import numpy as np
from os import listdir, path
import tempfile
import warnings

from art.utils import Deprecated

logger = logging.getLogger(__name__)


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


def test_predict(get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True)

    if classifier is not None:
        y_predicted = classifier.predict(x_test_mnist[0:1])
        y_expected = np.asarray(
            [
                [
                    0.12109935,
                    0.0498215,
                    0.0993958,
                    0.06410097,
                    0.11366927,
                    0.04645343,
                    0.06419806,
                    0.30685693,
                    0.07616713,
                    0.05823758,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


def test_nb_classes(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    if classifier is not None:
        assert classifier.nb_classes == 10


def test_input_shape(framework, get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    if classifier is not None:
        if framework == "pytorch":
            assert classifier.input_shape == (1, 28, 28)
        else:
            assert classifier.input_shape == (28, 28, 1)


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
                        "channel_index={Deprecated}, channels_first=False, clip_values=array([0., 1.], dtype=float32),"
                        "preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1))",
                    ]
            else:
                message_list = ["TensorFlowV2Classifier",
                                "model=",
                                "nb_classes=10",
                                "input_shape=(28, 28, 1)",
                                "loss_object=<tensorflow.python.keras.losses." "SparseCategoricalCrossentropy",
                                "train_step=<function get_image_classifier_tf_v2." "<locals>.train_step",
                                "channel_index={Deprecated}, channels_first=False, ",
                                "clip_values=array([0., 1.], dtype=float32), "
                                "preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1))",
                                ],

            if message_list is None:
                raise NotImplementedError(
                    "fw_agnostic_backend_test_repr not implemented for framework {0}".format(framework))

            repr_ = repr(classifier)
            for message in message_list:
                assert message in repr_, "{0}: was not contained within repr".format(message)

    except NotImplementedError as e:
        warnings.warn(UserWarning(e))
