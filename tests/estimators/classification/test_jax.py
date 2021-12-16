import logging

import numpy as np
import pytest
from jax import random
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from art.experimental.estimators.classification.jax import JaxClassifier
from tests.utils import ARTTestException


logger = logging.getLogger(__name__)


# Define a jax model
def jax_classifier():
    # Define model sizes
    layer_sizes = [784, 256, 128, 10]

    # Initialize random weights
    def random_init(m, n, k):
        w, b = random.split(k)
        rw, rb = random.normal(w, (n, m)) * 0.01, random.normal(b, (n,)) * 0.01
        return rw, rb

    model = [
        random_init(m, n, k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], random.split(random.PRNGKey(0), 4))
    ]

    # Forward function
    def forward(model, x):
        activation = jnp.reshape(x, 784)
        for w, b in model[:-1]:
            output = jnp.dot(w, activation) + b
            activation = jnp.maximum(0, output)

        f_w, f_b = model[-1]
        logit = jnp.dot(f_w, activation) + f_b
        return logit - logsumexp(logit)

    # Prediction function
    predict_func = vmap(forward, in_axes=(None, 0))

    # Loss function
    def loss_func(model, x, y):
        preds = predict_func(model, x)
        return -jnp.mean(preds * y)

    # Update function
    @jit
    def update_func(model, x, y):
        grads = grad(loss_func)(model, x, y)
        return [(w - 0.01 * dw, b - 0.01 * db) for (w, b), (dw, db) in zip(model, grads)]

    classifier = JaxClassifier(
        model=model,
        predict_func=predict_func,
        loss_func=loss_func,
        update_func=update_func,
        input_shape=(28, 28, 1),
        nb_classes=10,
    )

    return classifier


classifier = jax_classifier()


@pytest.mark.skip_framework("pytorch", "tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_fit(art_warning, get_default_mnist_subset, default_batch_size):
    try:
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        labels = np.argmax(y_test_mnist, axis=1)

        accuracy = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
        np.testing.assert_array_almost_equal(accuracy, 0.32, decimal=0.06)

        classifier.fit(x_train_mnist, y_train_mnist, batch_size=default_batch_size, nb_epochs=2)
        accuracy_2 = np.sum(np.argmax(classifier.predict(x_test_mnist), axis=1) == labels) / x_test_mnist.shape[0]
        np.testing.assert_array_almost_equal(accuracy_2, 0.73, decimal=0.06)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch", "tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_predict(art_warning, get_default_mnist_subset, expected_values):
    try:
        (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        y_predicted = classifier.predict(x_test_mnist[0:1])

        np.testing.assert_array_almost_equal(y_predicted, expected_values(), decimal=4)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch", "tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_shapes(art_warning, get_default_mnist_subset):
    try:
        (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        predictions = classifier.predict(x_test_mnist)

        assert predictions.shape == y_test_mnist.shape

        assert classifier.nb_classes == 10

        loss_gradients = classifier.loss_gradient(x_test_mnist[:11], y_test_mnist[:11])

        assert loss_gradients.shape == x_test_mnist[:11].shape

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch", "tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_loss_gradient(art_warning, get_default_mnist_subset, expected_values, mnist_shape):
    try:
        (expected_gradients_1, expected_gradients_2) = expected_values()

        (_, _), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        gradients = classifier.loss_gradient(x_test_mnist, y_test_mnist)

        assert gradients.shape == (x_test_mnist.shape[0],) + mnist_shape

        sub_gradients = gradients[0, 0, :, 0]
        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_gradients_1[0],
            decimal=expected_gradients_1[1],
        )

        sub_gradients = gradients[0, :, 14, 0]
        np.testing.assert_array_almost_equal(
            sub_gradients,
            expected_gradients_2[0],
            decimal=expected_gradients_2[1],
        )

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch", "tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_nb_classes(art_warning):
    try:
        assert classifier.nb_classes == 10

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch", "tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_input_shape(art_warning, mnist_shape):
    try:
        assert classifier.input_shape == mnist_shape

    except ARTTestException as e:
        art_warning(e)
