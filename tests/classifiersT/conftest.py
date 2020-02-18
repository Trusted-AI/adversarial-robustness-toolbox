import pytest
import numpy as np


@pytest.fixture
def backend_test_repr(get_image_classifier_list):
    classifier_list, _ = get_image_classifier_list()

    def _backend_test_repr(message_list):
        repr_ = repr(classifier_list[0])
        for message in message_list:
            assert message in repr_

    yield _backend_test_repr


@pytest.fixture
def get_backend_test_layers(get_mlFramework, get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier_list, _ = get_image_classifier_list()
    def _backend_test_layers(batch_size, layer_count=None):
        if layer_count is not None:
            assert len(classifier_list[0].layer_names) == layer_count

        for i, name in enumerate(classifier_list[0].layer_names):
            activation_i = classifier_list[0].get_activations(x_test_mnist, i, batch_size=batch_size)
            activation_name = classifier_list[0].get_activations(x_test_mnist, name, batch_size=batch_size)
            np.testing.assert_array_equal(activation_name, activation_i)
    yield _backend_test_layers