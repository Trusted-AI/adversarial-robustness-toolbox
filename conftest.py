import logging
import pytest
from art import utils
from tests import utils_test
import numpy as np
from art.defences import FeatureSqueezing
from art.classifiers import KerasClassifier

from art.classifiers.scikitlearn import SklearnClassifier


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
def fix_load_iris_dataset():
    logging.info("Loading Iris dataset")
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris), _, _ = utils.load_dataset('iris')

    yield (x_train_iris, y_train_iris), (x_test_iris, y_test_iris)

@pytest.fixture(scope="function")
def fix_get_iris(fix_load_iris_dataset, fix_mlFramework):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = fix_load_iris_dataset

    x_train_iris_original = x_train_iris.copy()
    y_train_iris_original = y_train_iris.copy()
    x_test_iris_original = x_test_iris.copy()
    y_test_iris_original = y_test_iris.copy()

    yield (x_train_iris, y_train_iris), (x_test_iris, y_test_iris)

    np.testing.assert_array_almost_equal(x_train_iris_original, x_train_iris, decimal=3)
    np.testing.assert_array_almost_equal(y_train_iris_original, y_train_iris, decimal=3)
    np.testing.assert_array_almost_equal(x_test_iris_original, x_test_iris, decimal=3)
    np.testing.assert_array_almost_equal(y_test_iris_original, y_test_iris, decimal=3)


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

@pytest.fixture
def make_customer_record():
    def _make_customer_record(name):
        return {"name": name, "orders": []}

    return _make_customer_record

@pytest.fixture
def tabular_classifier_list(fix_mlFramework):
    def _tabular_classifier_list(attack, clipped=True):
        if fix_mlFramework == "keras" and clipped is False:
            classifier = utils_test.get_tabular_classifier_kr()
            classifier_list =  [KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)]

        if fix_mlFramework == "tensorflow" and clipped is False:
            logging.warning("{0} doesn't have an uncliped classifier defined yet".format(fix_mlFramework))
            classifier_list =  None

        if fix_mlFramework == "pytorch" and clipped is False:
            logging.warning("{0} doesn't have an uncliped classifier defined yet".format(fix_mlFramework))
            classifier_list =  None

        if fix_mlFramework == "scikitlearn" and clipped is False:
            model_list = utils_test.get_tabular_classifier_scikit_list()
            classifier_list =  [SklearnClassifier(model=model) for model in model_list]

        if fix_mlFramework == "keras" and clipped is True:
            classifier_list =  [utils_test.get_tabular_classifier_kr()]

        if fix_mlFramework == "tensorflow" and clipped is True:
            classifier, _ = utils_test.get_tabular_classifier_tf()
            classifier_list = [classifier]

        if fix_mlFramework == "pytorch" and clipped is True:
            classifier_list =  [utils_test.get_tabular_classifier_pt()]

        if fix_mlFramework == "scikitlearn" and clipped is True:
            model_list = utils_test.get_tabular_classifier_scikit_list()
            classifier_list =  [SklearnClassifier(model=model, clip_values=(0, 1)) for model in model_list]

        if classifier_list is None: return None

        return [potential_classier for potential_classier in classifier_list if attack.is_valid_classifier_type(classifier)]


    return _tabular_classifier_list


@pytest.fixture
def unclipped_tabular_classifier_list(fix_mlFramework):
    if fix_mlFramework == "keras":
        classifier = utils_test.get_tabular_classifier_kr()
        return [KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)]

    if fix_mlFramework == "tensorflow":
        logging.warning("{0} doesn't have an uncliped classifier defined yet".format(fix_mlFramework))
        return None

    if fix_mlFramework == "pytorch":
        logging.warning("{0} doesn't have an uncliped classifier defined yet".format(fix_mlFramework))
        return None

    if fix_mlFramework == "scikitlearn":
        model_list = utils_test.get_tabular_classifier_scikit_list()
        return [SklearnClassifier(model=model) for model in model_list]

    raise Exception("A classifier factory method needs to be implemented for framework {0}".format(fix_mlFramework))

@pytest.fixture
def clipped_tabular_classifier_list(fix_mlFramework):
    if fix_mlFramework == "keras":
        return [utils_test.get_tabular_classifier_kr()]

    if fix_mlFramework == "tensorflow":
        classifier, _ = utils_test.get_tabular_classifier_tf()
        return [classifier]

    if fix_mlFramework == "pytorch":
        return [utils_test.get_tabular_classifier_pt()]

    if fix_mlFramework == "scikitlearn":
        model_list = utils_test.get_tabular_classifier_scikit_list()
        return [SklearnClassifier(model=model, clip_values=(0, 1)) for model in model_list]

    raise Exception("A classifier factory method needs to be implemented for framework {0}".format(fix_mlFramework))
