import logging
import numpy as np
from os import listdir, path
import pytest
import tempfile
import warnings

from art.utils import Deprecated

from tests.utils import ExpectedValue

logger = logging.getLogger(__name__)


# def test_fit_kwargs(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
#     (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
#
#     def get_lr(_):
#         return 0.01
#
#     # Test a valid callback
#     classifier, _ = get_image_classifier_list(one_classifier=True)
#     # kwargs = {"callbacks": [LearningRateScheduler(get_lr)]}
#     # classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=1, **kwargs)
#
#     # Test failure for invalid parameters
#     kwargs = {"epochs": 1}
#     # TODO only throws a TypeError in Keras, not in pytorch or tensorflow
#     with pytest.raises(TypeError) as exception:
#         try:
#             classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=1, **kwargs)
#             tmp1 = ""
#         except Exception as e:
#             tmp = ""
#
#     assert "multiple values for keyword argument" in str(exception.value)


def test_fit(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    labels = np.argmax(y_test_mnist, axis=1)
    classifier, sess = get_image_classifier_list(one_classifier=True)
    accuracy = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(accuracy, 0.32, decimal=0.06)

    classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=2)
    accuracy_2 = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(accuracy_2, 0.73, decimal=0.06)


def test_shapes(get_default_mnist_subset, get_image_classifier_list):
    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, sess = get_image_classifier_list(one_classifier=True)

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

    expected_values = {"pre_fit_accuracy": ExpectedValue(0.32, 0.06), "post_fit_accuracy": ExpectedValue(0.68, 0.06)}

    data_gen = image_data_generator(sess=sess)

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

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

    classifier.fit_generator(generator=data_gen, nb_epochs=2)
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


def test_loss_gradient(framework, is_tf_version_2, get_default_mnist_subset, get_image_classifier_list):
    expected_values = {
        "expected_gradients_1": ExpectedValue(
            np.asarray([
                0.00210803,
                0.00213919,
                0.0052098,
                0.00548,
                -0.0023059,
                0.00432076,
                0.00274945,
                0.,
                0.,
                -0.0583441,
                -0.00616604,
                0.05262191,
                -0.02373985,
                0.05273107,
                0.10711591,
                0.12773865,
                0.06892889,
                0.01337799,
                0.1003202,
                0.01681095,
                -0.00028647,
                -0.05588859,
                0.01474165,
                0.,
                0.,
                0.,
                0.,
                0.,
            ]),
            4,
        ),
        "expected_gradients_2": ExpectedValue(
            np.asarray(
                [0.0559206,
                 0.05338925,
                 0.0648919,
                 0.07925165,
                 -0.04029291,
                 - 0.11281465,
                 0.01850601,
                 0.00325053,
                 0.08163194,
                 0.03333949,
                 0.031766,
                 -0.02420464,
                 -0.07815557,
                 -0.04698735,
                 0.10711591,
                 0.04086433,
                 -0.03441072,
                 0.01071284,
                 - 0.04229196,
                 - 0.01386157,
                 0.02827487,
                 0.,
                 0.,
                 0.,
                 0.,
                 0.,
                 0.,
                 0.]),
            4,
        ),
    }

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    # Test gradient
    # x_test_mnist = x_test_mnist[:3]
    # y_test_mnist = y_test_mnist[:3]
    gradients = classifier.loss_gradient(x_test_mnist, y_test_mnist)

    if framework == "pytorch":
        assert gradients.shape == (x_test_mnist.shape[0], 1, 28, 28)
    else:
        assert gradients.shape == (x_test_mnist.shape[0], 28, 28, 1)

    if framework == "pytorch":
        sub_gradients = gradients[0, 0, :, 14]
    else:
        sub_gradients = gradients[0, :, 14, 0]

    if "expected_gradients_1" in expected_values:
        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1"].value,
            decimal=expected_values["expected_gradients_1"].decimals,
        )

    if framework == "pytorch":
        sub_gradients = gradients[0, 0, 14, :]
    else:
        sub_gradients = gradients[0, 14, :, 0]

    if "expected_gradients_2" in expected_values:
        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_2"].value,
            decimal=expected_values["expected_gradients_2"].decimals,
        )


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

    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    if classifier is not None:
        y_predicted = classifier.predict(x_test_mnist[0:1])

        y_expected = np.asarray(
            [
                [0.15710345,
                 - 0.73106134,
                 - 0.04039804,
                 - 0.47904843,
                 0.09378531,
                 - 0.80105764,
                 - 0.47753483,
                 1.0868737,
                 - 0.3065778,
                 - 0.57497704]
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
                        f"channel_index={Deprecated}, channels_first=False, ",
                        "clip_values=array([0., 1.], dtype=float32), ",
                        "preprocessing_defences=None, postprocessing_defences=None, preprocessing=(0, 1))",
                    ]

                else:

                    message_list = [
                        "TensorFlowClassifier",
                        "input_ph=<tf.Tensor 'Placeholder:0' shape=(?, 28, 28, 1) dtype=float32>",
                        "output=<tf.Tensor 'dense/BiasAdd:0' shape=(?, 10) dtype=float32",
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


def test_class_gradient(framework, get_image_classifier_list, get_default_mnist_subset):
    expected_values = {
        "expected_gradients_1_all_labels": ExpectedValue(
            np.asarray(
                [
                    -0.03347399,
                    -0.03195872,
                    -0.02650188,
                    0.04111874,
                    0.08676253,
                    0.03339913,
                    0.06925241,
                    0.09387045,
                    0.15184258,
                    -0.00684002,
                    0.05070481,
                    0.01409407,
                    -0.03632583,
                    0.00151133,
                    0.05102589,
                    0.00766463,
                    -0.00898967,
                    0.00232938,
                    -0.00617045,
                    -0.00201032,
                    0.00410065,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_2_all_labels": ExpectedValue(
            np.asarray(
                [
                    -0.09723657,
                    -0.00240533,
                    0.02445251,
                    -0.00035474,
                    0.04765627,
                    0.04286841,
                    0.07209076,
                    0.0,
                    0.0,
                    -0.07938144,
                    -0.00142567,
                    0.02882954,
                    -0.00049514,
                    0.04170151,
                    0.05102589,
                    0.09544909,
                    -0.04401167,
                    -0.06158172,
                    0.03359772,
                    -0.00838454,
                    0.01722163,
                    -0.13376027,
                    0.08206709,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_1_label5": ExpectedValue(
            np.asarray(
                [
                    -0.03347399,
                    -0.03195872,
                    -0.02650188,
                    0.04111874,
                    0.08676253,
                    0.03339913,
                    0.06925241,
                    0.09387045,
                    0.15184258,
                    -0.00684002,
                    0.05070481,
                    0.01409407,
                    -0.03632583,
                    0.00151133,
                    0.05102589,
                    0.00766463,
                    -0.00898967,
                    0.00232938,
                    -0.00617045,
                    -0.00201032,
                    0.00410065,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_2_label5": ExpectedValue(
            np.asarray(
                [
                    -0.09723657,
                    -0.00240533,
                    0.02445251,
                    -0.00035474,
                    0.04765627,
                    0.04286841,
                    0.07209076,
                    0.0,
                    0.0,
                    -0.07938144,
                    -0.00142567,
                    0.02882954,
                    -0.00049514,
                    0.04170151,
                    0.05102589,
                    0.09544909,
                    -0.04401167,
                    -0.06158172,
                    0.03359772,
                    -0.00838454,
                    0.01722163,
                    -0.13376027,
                    0.08206709,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_1_labelArray": ExpectedValue(
            np.asarray(
                [
                    0.06860766,
                    0.065502,
                    0.08539103,
                    0.13868105,
                    -0.05520725,
                    -0.18788849,
                    0.02264893,
                    0.02980516,
                    0.2226511,
                    0.11288887,
                    -0.00678776,
                    0.02045561,
                    -0.03120914,
                    0.00642691,
                    0.08449504,
                    0.02848018,
                    -0.03251382,
                    0.00854315,
                    -0.02354656,
                    -0.00767687,
                    0.01565931,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_2_labelArray": ExpectedValue(
            np.asarray(
                [
                    -0.0487146,
                    -0.0171556,
                    -0.03161772,
                    -0.0420007,
                    0.03360246,
                    -0.01864819,
                    0.00315916,
                    0.0,
                    0.0,
                    -0.07631349,
                    -0.00374462,
                    0.04229517,
                    -0.01131879,
                    0.05044588,
                    0.08449504,
                    0.12417868,
                    0.07536847,
                    0.03906382,
                    0.09467953,
                    0.00543209,
                    -0.00504872,
                    -0.03366479,
                    -0.00385999,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
    }

    # labels = np.random.randint(5, size=x_test_mnist.shape[0])
    labels = np.asarray(
        [
            3,
            4,
            4,
            0,
            1,
            1,
            1,
            2,
            3,
            4,
            4,
            2,
            2,
            0,
            0,
            4,
            0,
            1,
            2,
            0,
            3,
            4,
            2,
            2,
            3,
            3,
            0,
            1,
            3,
            0,
            3,
            2,
            3,
            4,
            1,
            3,
            3,
            3,
            2,
            1,
            3,
            4,
            2,
            3,
            4,
            1,
            4,
            0,
            4,
            1,
            1,
            4,
            1,
            4,
            0,
            1,
            0,
            0,
            4,
            0,
            4,
            2,
            3,
            1,
            2,
            2,
            4,
            3,
            4,
            2,
            2,
            4,
            4,
            2,
            1,
            3,
            2,
            1,
            4,
            1,
            0,
            1,
            2,
            1,
            2,
            1,
            2,
            1,
            1,
            4,
            1,
            2,
            4,
            0,
            4,
            1,
            2,
            1,
            1,
            3,
        ]
    )

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
    # Test all gradients label
    gradients = classifier.class_gradient(x_test_mnist)

    if framework == "pytorch":
        assert gradients.shape == (x_test_mnist.shape[0], 10, 1, 28, 28)
    else:
        assert gradients.shape == (x_test_mnist.shape[0], 10, 28, 28, 1)

    if "expected_gradients_1_all_labels" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 5, 0, 14, :]  # expected_gradients_1_all_labels
        else:
            sub_gradients = gradients[0, 5, 14, :, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1_all_labels"].value,
            decimal=expected_values["expected_gradients_1_all_labels"].decimals,
        )

    if "expected_gradients_2_all_labels" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 5, 0, :, 14]  # expected_gradients_2_all_labels
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
            sub_gradients = gradients[0, 0, 0, 14, :]  # expected_gradients_1_label5
        else:
            sub_gradients = gradients[0, 0, 14, :, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1_label5"].value,
            decimal=expected_values["expected_gradients_1_label5"].decimals,
        )

    if "expected_gradients_2_label5" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, :, 14]  # expected_gradients_2_all_labels
        else:
            sub_gradients = gradients[0, 0, :, 14, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_2_label5"].value,
            decimal=expected_values["expected_gradients_2_label5"].decimals,
        )

    # # Test a set of gradients label = array
    # # label = np.random.randint(5, size=self.n_test)
    gradients = classifier.class_gradient(x_test_mnist, label=labels)

    if framework == "pytorch":
        assert gradients.shape == (x_test_mnist.shape[0], 1, 1, 28, 28)
    else:
        assert gradients.shape == (x_test_mnist.shape[0], 1, 28, 28, 1)

    if "expected_gradients_1_labelArray" in expected_values:

        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, 14, :]  # expected_gradients_1_labelArray
        else:
            sub_gradients = gradients[0, 0, 14, :, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_1_labelArray"].value,
            decimal=expected_values["expected_gradients_1_labelArray"].decimals,
        )

    if "expected_gradients_2_labelArray" in expected_values:
        if framework == "pytorch":
            sub_gradients = gradients[0, 0, 0, :, 14]  # expected_gradients_2_labelArray
        else:
            sub_gradients = gradients[0, 0, :, 14, 0]

        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_values["expected_gradients_2_labelArray"].value,
            decimal=expected_values["expected_gradients_2_labelArray"].decimals,
        )
