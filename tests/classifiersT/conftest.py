import pytest
import numpy as np

@pytest.fixture
def get_backend_test_layers(get_mlFramework, get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier = get_image_classifier_list()[0]
    def _backend_test_layers(batch_size, layer_count=None):
        if layer_count is not None:
            assert len(classifier.layer_names) == layer_count

        for i, name in enumerate(classifier.layer_names):
            activation_i = classifier.get_activations(x_test_mnist, i, batch_size=batch_size)
            activation_name = classifier.get_activations(x_test_mnist, name, batch_size=batch_size)
            np.testing.assert_array_equal(activation_name, activation_i)
    yield _backend_test_layers