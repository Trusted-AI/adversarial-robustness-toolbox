import logging
import pytest
from art import utils
from tests import utils_test
import numpy as np
from art.classifiers import KerasClassifier
import tensorflow as tf
import os
import requests
import tempfile
import shutil
from tests import utils_test
from art.defences import FeatureSqueezing
from art.classifiers import KerasClassifier

logger = logging.getLogger(__name__)
art_supported_frameworks = ["keras", "tensorflow", "pytorch", "scikitlearn"]

utils.master_seed(1234)

def pytest_addoption(parser):
    parser.addoption(
        "--mlFramework", action="store", default="tensorflow", help="ART tests allow you to specify which mlFramework to use. The default mlFramework used is tensorflow. Other options available are {0}".format(art_supported_frameworks)
    )


@pytest.fixture
def get_image_classifier_list(get_mlFramework):
    def _get_image_classifier_list(defended=False):
        if get_mlFramework == "keras":
            if defended:
                classifier = utils_test.get_image_classifier_kr()
                # Get the ready-trained Keras model
                fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
                classifier_list = [KerasClassifier(model=classifier._model, clip_values=(0, 1), defences=fs)]
            else:
                classifier_list = [utils_test.get_image_classifier_kr()]
        if get_mlFramework == "tensorflow":
            if defended:
                logging.warning("{0} doesn't have a defended image classifier defined yet".format(get_mlFramework))
                classifier_list = None
            else:
                classifier, sess = utils_test.get_image_classifier_tf()
                classifier_list = [classifier]
        if get_mlFramework == "pytorch":
            if defended:
                logging.warning("{0} doesn't have a defended image classifier defined yet".format(get_mlFramework))
                classifier_list = None
            else:
                classifier_list = [utils_test.get_image_classifier_pt()]
        if get_mlFramework == "scikitlearn":
            if defended:
                logging.warning("{0} doesn't have a defended image classifier defined yet".format(get_mlFramework))
                classifier_list = None
            else:
                logging.warning("{0} doesn't have an image classifier defined yet".format(get_mlFramework))
                classifier_list = None

        if classifier_list is None:
            return None

        return classifier_list

    return _get_image_classifier_list

@pytest.fixture(scope="function")
def create_test_image(create_test_dir):
    test_dir = create_test_dir
    # Download one ImageNet pic for tests
    url = 'http://farm1.static.flickr.com/163/381342603_81db58bea4.jpg'
    result = requests.get(url, stream=True)
    if result.status_code == 200:
        image = result.raw.read()
        f = open(os.path.join(test_dir, 'test.jpg'), 'wb')
        f.write(image)
        f.close()

    yield os.path.join(test_dir, 'test.jpg')

@pytest.fixture
def get_mlFramework(request):
    mlFramework = request.config.getoption("--mlFramework")
    if mlFramework not in art_supported_frameworks:
        raise Exception("mlFramework value {0} is unsupported. Please use one of these valid values: {1}".format(
            mlFramework, " ".join(art_supported_frameworks)))
    # if utils_test.is_valid_framework(mlFramework):
    #     raise Exception("The mlFramework specified was incorrect. Valid options available are {0}".format(art_supported_frameworks))
    return mlFramework


@pytest.fixture(scope="session")
def default_batch_size():
    yield 16

@pytest.fixture(scope="session")
def is_tf_version_2():
    if tf.__version__[0] == '2':
        yield True
    else:
        yield False

@pytest.fixture(scope="session")
def load_iris_dataset():
    logging.info("Loading Iris dataset")
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris), _, _ = utils.load_dataset('iris')

    yield (x_train_iris, y_train_iris), (x_test_iris, y_test_iris)

@pytest.fixture(scope="function")
def get_iris_dataset(load_iris_dataset, get_mlFramework):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = load_iris_dataset

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
def default_dataset_subset_sizes():
    n_train = 1000
    n_test = 100
    yield n_train, n_test

@pytest.fixture()
def get_default_mnist_subset(get_mnist_dataset, default_dataset_subset_sizes):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train, n_test = default_dataset_subset_sizes

    yield (x_train_mnist[:n_train], y_train_mnist[:n_train]), (x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.fixture(scope="session")
def load_mnist_dataset():
    logging.info("Loading mnist")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = utils.load_dataset('mnist')
    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)

@pytest.fixture(scope="function")
def create_test_dir():
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture(scope="function")
def get_mnist_dataset(load_mnist_dataset, get_mlFramework):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist_dataset

    if get_mlFramework == "pytorch":
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




# ART test fixture to skip test for specific mlFramework values
# eg: @pytest.mark.only_with_platform("tensorflow")
@pytest.fixture(autouse=True)
def only_with_platform(request, get_mlFramework):
    if request.node.get_closest_marker('only_with_platform'):
        if get_mlFramework not in request.node.get_closest_marker('only_with_platform').args:
            pytest.skip('skipped on this platform: {}'.format(get_mlFramework))

# ART test fixture to skip test for specific mlFramework values
# eg: @pytest.mark.skipMlFramework("tensorflow","scikitlearn")
@pytest.fixture(autouse=True)
def skip_by_platform(request, get_mlFramework):
    if request.node.get_closest_marker('skipMlFramework'):
        if get_mlFramework in request.node.get_closest_marker('skipMlFramework').args:
            pytest.skip('skipped on this platform: {}'.format(get_mlFramework))

@pytest.fixture
def make_customer_record():
    def _make_customer_record(name):
        return {"name": name, "orders": []}

    return _make_customer_record

@pytest.fixture
def get_tabular_classifier_list(get_mlFramework):
    def _tabular_classifier_list(attack, clipped=True):
        if get_mlFramework == "keras":
            if clipped:
                classifier_list = [utils_test.get_tabular_classifier_kr()]
            else:
                classifier = utils_test.get_tabular_classifier_kr()
                classifier_list = [KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)]

        if get_mlFramework == "tensorflow":
            if clipped:
                classifier, _ = utils_test.get_tabular_classifier_tf()
                classifier_list = [classifier]
            else:
                logging.warning("{0} doesn't have an uncliped classifier defined yet".format(get_mlFramework))
                classifier_list =  None

        if get_mlFramework == "pytorch":
            if clipped:
                classifier_list = [utils_test.get_tabular_classifier_pt()]
            else:
                logging.warning("{0} doesn't have an uncliped classifier defined yet".format(get_mlFramework))
                classifier_list = None

        if get_mlFramework == "scikitlearn":
            return utils_test.get_tabular_classifier_scikit_list(clipped=False)
        if classifier_list is None:
            return None

        return [potential_classier for potential_classier in classifier_list if attack.is_valid_classifier_type(potential_classier)]


    return _tabular_classifier_list

