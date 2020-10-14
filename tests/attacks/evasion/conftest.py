import logging

from sklearn.datasets import load_digits
import numpy as np
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def _load_uci_digits_dataset():
    # TODO this should be removed. Tests using this fixture should use the mnist dataset instead
    logging.info("Loading UCI Digits Dataset")
    digits = load_digits()
    x_train = digits.data
    y_train = digits.target
    yield x_train, y_train


@pytest.fixture(scope="function")
def uci_digit_dataset(_load_uci_digits_dataset):
    #TODO this should be removed. Tests using this fixture should use the mnist dataset instead
    x_train, y_train = _load_uci_digits_dataset

    x_train_original = x_train.copy()
    y_train_original = y_train.copy()

    yield x_train, y_train

    # Check that the test data has not been modified, only catches changes in attack.generate if self has been used
    np.testing.assert_array_almost_equal(x_train_original, x_train, decimal=3)
    np.testing.assert_array_almost_equal(y_train_original, y_train, decimal=3)
