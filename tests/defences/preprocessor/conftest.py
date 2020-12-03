import numpy as np
import pytest


@pytest.fixture
def image_batch(channels_first):
    """
    Image fixture of shape NHWC and NCHW.
    """
    test_input = np.repeat(np.array(range(6)).reshape(6, 1), 24, axis=1).reshape((2, 3, 4, 6))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def image_batch_small():
    """Create image fixture of shape (batch_size, channels, width, height)."""
    return np.zeros((2, 1, 4, 4))


@pytest.fixture
def video_batch(channels_first):
    """
    Video fixture of shape NFHWC and NCFHW.
    """
    test_input = np.repeat(np.array(range(6)).reshape(6, 1), 24, axis=1).reshape((1, 3, 2, 4, 6))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 4, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def tabular_batch():
    """
    Create tabular data fixture of shape (batch_size, features).
    """
    return np.zeros((2, 4))
