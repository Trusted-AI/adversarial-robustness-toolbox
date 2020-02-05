import logging
import pytest
from art import utils
from tests import utils_test
import numpy as np
from art.defences import FeatureSqueezing
from art.classifiers import KerasClassifier

logger = logging.getLogger(__name__)
art_supported_frameworks = ["keras", "tensorflow", "pytorch", "scikitlearn"]

def pytest_addoption(parser):
    parser.addoption(
        "--mlFramework", action="store", default="tensorflow", help="ART tests allow you to specify which mlFramework to use. The default mlFramework used is tensorflow. Other options available are {0}".format(art_supported_frameworks)
    )

@pytest.fixture
def fix_mlFramework(request):
    mlFramework = request.config.getoption("--mlFramework")
    if mlFramework not in art_supported_frameworks:
        raise Exception("mlFramework value {0} is unsupported. Please use one of these valid values: {1}".format(
            mlFramework, " ".join(art_supported_frameworks)))
    # if utils_test.is_valid_framework(mlFramework):
    #     raise Exception("The mlFramework specified was incorrect. Valid options available are {0}".format(art_supported_frameworks))
    return mlFramework

@pytest.fixture(scope="session")
def fix_load_mnist_dataset():
    logging.info("Loading mnist")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = utils.load_dataset('mnist')
    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)

@pytest.fixture(scope="function")
def fix_get_mnist(fix_load_mnist_dataset, fix_mlFramework):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = fix_load_mnist_dataset

    if fix_mlFramework == "pytorch":
        x_test_mnist = np.reshape(x_test_mnist, (x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)

    x_train_mnist_original = x_train_mnist.copy()
    y_train_mnist_original = y_train_mnist.copy()
    x_test_mnist_original = x_test_mnist.copy()
    y_test_mnist_original = y_test_mnist.copy()

    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)

    # Check that the test data has not been modified, only catches changes in attack.generate if self has been used
    np.testing.assert_array_almost_equal(x_train_mnist_original, x_train_mnist, decimal=3)
    np.testing.assert_array_almost_equal(y_train_mnist_original, y_train_mnist, decimal=3)
    np.testing.assert_array_almost_equal(x_test_mnist_original, x_test_mnist, decimal=3)
    np.testing.assert_array_almost_equal(y_test_mnist_original, y_test_mnist, decimal=3)


@pytest.fixture
def defended_image_classifier_list(fix_mlFramework):
    if fix_mlFramework == "keras":
        classifier = utils_test.get_image_classifier_kr()
        # Get the ready-trained Keras model
        fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
        return [KerasClassifier(model=classifier._model, clip_values=(0, 1), defences=fs)]
    else:
        logging.warning("{0} doesn't have a defended image classifier defined yet".format(fix_mlFramework))
        return None

@pytest.fixture
def image_classifier_list(fix_mlFramework):
    if fix_mlFramework == "keras":
        return [utils_test.get_image_classifier_kr()]
    if fix_mlFramework == "tensorflow":
        classifier, sess = utils_test.get_image_classifier_tf()
        return [classifier]
    if fix_mlFramework == "pytorch":
        return [utils_test.get_image_classifier_pt()]
    if fix_mlFramework == "scikitlearn":
        logging.warning("{0} doesn't have an image classifier defined yet".format(fix_mlFramework))
        return None

    raise Exception("A classifier factory method needs to be implemented for framework {0}".format(fix_mlFramework))
