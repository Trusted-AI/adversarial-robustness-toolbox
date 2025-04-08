import pytest
from art.attacks.poisoning.perturbations.network_perturbations import *  # Assuming your file is named network_perturbations.py
from art.attacks.poisoning.perturbations.network_perturbations import _flip_target_label


# Initialize the array once for all tests
@pytest.fixture(scope="module")
def sample_array():
    return np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

@pytest.mark.framework_agnostic
@pytest.mark.flip_target_tests
def test_100_percent(sample_array):
    result = _flip_target_label(sample_array, flip_target=1, poison_percentage=1.0)
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.framework_agnostic
@pytest.mark.flip_target_tests
def test_0_percent(sample_array):
    result = _flip_target_label(sample_array, flip_target=1, poison_percentage=0.0)
    np.testing.assert_array_equal(result, sample_array)

@pytest.mark.framework_agnostic
@pytest.mark.flip_target_tests
def test_bool_array(sample_array):
    bool_array = sample_array.astype(np.bool_)
    result = _flip_target_label(bool_array, flip_target=1, poison_percentage=0.5)
    assert result.dtype == np.bool_

@pytest.mark.framework_agnostic
@pytest.mark.flip_target_tests
def test_empty_array():
    empty_array = np.array([], dtype=np.int64)
    result = _flip_target_label(empty_array, flip_target=1, poison_percentage=0.5)
    np.testing.assert_array_equal(result, empty_array)
