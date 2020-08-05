import keras
import logging
import numpy as np
from os import listdir, path
import pytest
import tempfile
import warnings

logger = logging.getLogger(__name__)


def is_keras_2_3():
    if int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3:
        return True
    return False


def test_layers(get_default_mnist_subset, framework, is_tf_version_2, get_image_classifier_list):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
        if classifier is not None:
            (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

            if framework == "tensorflow" and is_tf_version_2:
                raise NotImplementedError(
                    "fw_agnostic_backend_test_layers not implemented for framework {0}".format(framework)
                )

            batch_size = 128
            for i, name in enumerate(classifier.layer_names):
                activation_i = classifier.get_activations(x_test_mnist, i, batch_size=batch_size)
                activation_name = classifier.get_activations(x_test_mnist, name, batch_size=batch_size)
                np.testing.assert_array_equal(activation_name, activation_i)
    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


# Note: because mxnet only supports 1 concurrent version of a model if we fit that model, all expected values will
# change for all other tests using that fitted model
@pytest.mark.skipMlFramework("mxnet", "scikitlearn")
def test_fit(get_default_mnist_subset, default_batch_size, get_image_classifier_list):
    try:
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        labels = np.argmax(y_test_mnist, axis=1)
        classifier, sess = get_image_classifier_list(one_classifier=True, from_logits=True)

        accuracy = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
        np.testing.assert_array_almost_equal(accuracy, 0.32, decimal=0.06)

        classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=2)
        accuracy_2 = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
        np.testing.assert_array_almost_equal(accuracy_2, 0.73, decimal=0.06)
    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


@pytest.mark.skipMlFramework("scikitlearn")
def test_predict(
        request, framework, get_default_mnist_subset, get_image_classifier_list, expected_values, store_expected_values
):
    if framework == "keras" and is_keras_2_3() is False:
        # Keras 2.2 does not support creating classifiers with logits=True so skipping this test
        return

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    if classifier is not None:
        y_predicted = classifier.predict(x_test_mnist[0:1])
        np.testing.assert_array_almost_equal(y_predicted, expected_values, decimal=4)


@pytest.mark.skipMlFramework("scikitlearn")
def test_shapes(get_default_mnist_subset, get_image_classifier_list):
    try:
        (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
        classifier, sess = get_image_classifier_list(one_classifier=True, from_logits=True)

        predictions = classifier.predict(x_test_mnist)
        assert predictions.shape == y_test_mnist.shape

        assert classifier.nb_classes == 10

        class_gradients = classifier.class_gradient(x_test_mnist[:11])
        assert class_gradients.shape == tuple([11, 10] + list(x_test_mnist[1].shape))

        loss_gradients = classifier.loss_gradient(x_test_mnist[:11], y_test_mnist[:11])
        assert loss_gradients.shape == x_test_mnist[:11].shape

    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


# Note: because mxnet only supports 1 concurrent version of a model if we fit that model, all expected values will
# change for all other tests using that fitted model
@pytest.mark.skipMlFramework("mxnet", "scikitlearn")
def test_fit_image_generator(
        framework, is_tf_version_2, get_image_classifier_list, image_data_generator, get_default_mnist_subset
):
    try:
        if framework == "tensorflow" and is_tf_version_2:
            return

        classifier, sess = get_image_classifier_list(one_classifier=True, from_logits=True)

        (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        true_class = np.argmax(y_test_mnist, axis=1)

        predictions = classifier.predict(x_test_mnist)
        prediction_class = np.argmax(predictions, axis=1)
        pre_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]

        np.testing.assert_array_almost_equal(
            pre_fit_accuracy, 0.32, decimal=0.06,
        )

        data_gen = image_data_generator(sess=sess)
        classifier.fit_generator(generator=data_gen, nb_epochs=2)
        predictions = classifier.predict(x_test_mnist)
        prediction_class = np.argmax(predictions, axis=1)
        post_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]

        np.testing.assert_array_almost_equal(
            post_fit_accuracy, 0.68, decimal=0.06,
        )
    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


@pytest.mark.skipMlFramework("scikitlearn")
def test_loss_gradient(
        framework,
        is_tf_version_2,
        get_default_mnist_subset,
        get_image_classifier_list,
        expected_values,
        mnist_shape,
        store_expected_values,
):
    if framework == "keras" and is_keras_2_3() is False:
        # Keras 2.2 does not support creating classifiers with logits=True so skipping this test d
        return

    (expected_gradients_1, expected_gradients_2) = expected_values

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    if classifier is not None:
        gradients = classifier.loss_gradient(x_test_mnist, y_test_mnist)

        assert gradients.shape == (x_test_mnist.shape[0],) + mnist_shape

        if mnist_shape[0] == 1:
            sub_gradients = gradients[0, 0, :, 14]
        else:
            sub_gradients = gradients[0, :, 14, 0]

        # store_1 = (sub_gradients.tolist(), expected_gradients_1[1])
        np.testing.assert_array_almost_equal(
            sub_gradients, expected_gradients_1[0], decimal=expected_gradients_1[1],
        )
        # np.testing.assert_array_almost_equal(
        #     sub_gradients, store_1[0], decimal=store_1[1],
        # )

        if mnist_shape[0] == 1:
            sub_gradients = gradients[0, 0, 14, :]
        else:
            sub_gradients = gradients[0, 14, :, 0]

        # store_2 = (sub_gradients.tolist(), expected_gradients_2[1])
        np.testing.assert_array_almost_equal(
            sub_gradients, expected_gradients_2[0], decimal=expected_gradients_2[1],
        )

        # np.testing.assert_array_almost_equal(
        #     sub_gradients, store_2[0], decimal=store_2[1],
        # )

        # store_values = (store_1, store_2)
        # store_expected_values(store_values, framework)


@pytest.mark.skipMlFramework("scikitlearn")
def test_nb_classes(get_image_classifier_list):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
        assert classifier.nb_classes == 10
    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


@pytest.mark.skipMlFramework("scikitlearn")
def test_input_shape(get_image_classifier_list, mnist_shape):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

        assert classifier.input_shape == mnist_shape
    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


def test_save(get_image_classifier_list):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
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


@pytest.mark.skipMlFramework("scikitlearn")
def test_repr(get_image_classifier_list, framework, expected_values, store_expected_values):
    try:
        classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
        if classifier is not None:

            repr_ = repr(classifier)
            for message in expected_values:
                assert message in repr_, "{0}: was not contained within repr".format(message)

    except NotImplementedError as e:
        warnings.warn(UserWarning(e))


@pytest.mark.skipMlFramework("scikitlearn")
def test_class_gradient(
        framework, get_image_classifier_list, get_default_mnist_subset, mnist_shape, store_expected_values,
        expected_values
):
    if framework == "keras" and is_keras_2_3() is False:
        # Keras 2.2 does not support creating classifiers with logits=True so skipping this test
        return

    (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
    if classifier is not None:

        (
            grad_1_all_labels,
            grad_2_all_labels,
            grad_1_label5,
            grad_2_label5,
            grad_1_labelArray,
            grad_2_labelArray,
            labels_list,
        ) = expected_values

        labels = np.array(labels_list, dtype=object)

        # TODO we should consider checking channel independent columns to make this test truly framework independent
        def get_gradient1_column(gradients):
            if mnist_shape[0] == 1:
                return gradients[0, 5, 0, 14, :]  # expected_gradients_1_all_labels
            else:
                return gradients[0, 5, 14, :, 0]

        def get_gradient2_column(gradients):
            if mnist_shape[0] == 1:
                return gradients[0, 5, 0, :, 14]  # expected_gradients_2_all_labels
            else:
                return gradients[0, 5, :, 14, 0]

        def get_gradient3_column(gradients):
            if mnist_shape[0] == 1:
                return gradients[0, 0, 0, 14, :]  # expected_gradients_1_label5
            else:
                return gradients[0, 0, 14, :, 0]

        def get_gradient4_column(gradients):
            if mnist_shape[0] == 1:
                return gradients[0, 0, 0, :, 14]  # expected_gradients_2_all_labels
            else:
                return gradients[0, 0, :, 14, 0]

        # Test all gradients label
        gradients = classifier.class_gradient(x_test_mnist)

        new_shape = (x_test_mnist.shape[0], 10,) + mnist_shape
        assert gradients.shape == new_shape

        sub_gradients1 = get_gradient1_column(gradients)

        np.testing.assert_array_almost_equal(sub_gradients1, grad_1_all_labels[0], decimal=4, )

        sub_gradients2 = get_gradient2_column(gradients)

        np.testing.assert_array_almost_equal(sub_gradients2, grad_2_all_labels[0], decimal=4, )

        # Test 1 gradient label = 5
        gradients = classifier.class_gradient(x_test_mnist, label=5)

        assert gradients.shape == (x_test_mnist.shape[0], 1,) + mnist_shape

        sub_gradients2 = get_gradient3_column(gradients)

        np.testing.assert_array_almost_equal(sub_gradients2, grad_1_label5[0], decimal=4, )

        sub_gradients4 = get_gradient4_column(gradients)

        np.testing.assert_array_almost_equal(sub_gradients4, grad_2_label5[0], decimal=4, )

        # # Test a set of gradients label = array
        gradients = classifier.class_gradient(x_test_mnist, label=labels)

        new_shape = (x_test_mnist.shape[0], 1,) + mnist_shape
        assert gradients.shape == new_shape

        sub_gradients5 = get_gradient3_column(gradients)

        np.testing.assert_array_almost_equal(sub_gradients5, grad_1_labelArray[0], decimal=4, )

        sub_gradients6 = get_gradient4_column(gradients)

        np.testing.assert_array_almost_equal(sub_gradients6, grad_2_labelArray[0], decimal=4, )



# TODO originally from pytorch
# def test_pickle(get_default_mnist_subset, get_image_classifier_list):
#     (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
#
#     classifier, _ = get_image_classifier_list(one_classifier=True, from_logits=True)
#     if classifier is not None:
#
#         from art.config import ART_DATA_PATH
#         import os, pickle
#
#         full_path = os.path.join(ART_DATA_PATH, "my_classifier")
#         folder = os.path.split(full_path)[0]
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#
#         # model = Model()
#         # loss_fn = nn.CrossEntropyLoss()
#         # optimizer = optim.Adam(model.parameters(), lr=0.01)
#         # myclassifier_2 = PyTorchClassifier(
#         #     model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
#         # )
#         classifier.fit(x_train_mnist, y_train_mnist, batch_size=100, nb_epochs=1)
#
#         pickle.dump(classifier, open(full_path, "wb"))
#
#         with open(full_path, "rb") as f:
#             loaded_model = pickle.load(f)
#             np.testing.assert_equal(classifier._clip_values, loaded_model._clip_values)
#             assert classifier._channel_index == loaded_model._channel_index
#             assert set(classifier.__dict__.keys()) == set(loaded_model.__dict__.keys())
#
#         # Test predict
#         predictions_1 = classifier.predict(x_test_mnist)
#         accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
#         predictions_2 = loaded_model.predict(x_test_mnist)
#         accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
#         assert accuracy_1 == accuracy_2
