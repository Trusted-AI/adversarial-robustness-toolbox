import logging
import pytest
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import requests
import tempfile
from torch.utils.data import DataLoader
import torch
import shutil
from tests.utils import master_seed, get_image_classifier_kr, get_image_classifier_tf, get_image_classifier_pt
from tests.utils import get_tabular_classifier_kr, get_tabular_classifier_tf, get_tabular_classifier_pt
from tests.utils import get_tabular_classifier_scikit_list, load_dataset
from art.data_generators import PyTorchDataGenerator, TensorFlowDataGenerator, KerasDataGenerator
from art.estimators.classification import KerasClassifier

logger = logging.getLogger(__name__)
art_supported_frameworks = ["keras", "tensorflow", "pytorch", "scikitlearn"]

master_seed(1234)

default_framework = "tensorflow"


def pytest_addoption(parser):
    parser.addoption(
        "--mlFramework", action="store", default=default_framework,
        help="ART tests allow you to specify which mlFramework to use. The default mlFramework used is tensorflow. "
             "Other options available are {0}".format(art_supported_frameworks)
    )


@pytest.fixture
def get_image_classifier_list_defended(framework):
    def _get_image_classifier_list_defended(one_classifier=False, **kwargs):
        sess = None
        classifier_list = None
        if framework == "keras":
            classifier = utils.get_image_classifier_kr()
            # Get the ready-trained Keras model
            fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
            classifier_list = [KerasClassifier(model=classifier._model, clip_values=(0, 1), preprocessing_defences=fs)]

        if framework == "tensorflow":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if framework == "pytorch":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if framework == "scikitlearn":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if classifier_list is None:
            return None, None

        if one_classifier:
            return classifier_list[0], sess

        return classifier_list, sess

    return _get_image_classifier_list_defended


@pytest.fixture
def get_image_classifier_list_for_attack(get_image_classifier_list, get_image_classifier_list_defended):
    def get_image_classifier_list_for_attack(attack, defended=False, **kwargs):
        if defended:
            classifier_list, _ = get_image_classifier_list_defended(kwargs)
        else:
            classifier_list, _ = get_image_classifier_list()
        if classifier_list is None:
            return None

        return [
            potential_classifier
            for potential_classifier in classifier_list
            if all(t in type(potential_classifier).__mro__ for t in attack._estimator_requirements)
        ]

    return get_image_classifier_list_for_attack


@pytest.fixture(autouse=True)
def setup_tear_down_framework(framework):
    # Ran before each test
    if framework == "keras":
        pass
    if framework == "tensorflow":
        # tf.reset_default_graph()
        if tf.__version__[0] != '2':
            tf.reset_default_graph()
    if framework == "pytorch":
        pass
    if framework == "scikitlearn":
        pass
    yield True

    # Ran after each test
    if framework == "keras":
        keras.backend.clear_session()
    if framework == "tensorflow":
        pass
    if framework == "pytorch":
        pass
    if framework == "scikitlearn":
        pass


@pytest.fixture
def image_iterator(framework, is_tf_version_2, get_default_mnist_subset, default_batch_size):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    if framework == "keras":
        keras_gen = ImageDataGenerator(
            width_shift_range=0.075,
            height_shift_range=0.075,
            rotation_range=12,
            shear_range=0.075,
            zoom_range=0.05,
            fill_mode="constant",
            cval=0,
        )
        return keras_gen.flow(x_train_mnist, y_train_mnist, batch_size=default_batch_size)

    if framework == "tensorflow":
        if not is_tf_version_2:
            x_tensor = tf.convert_to_tensor(x_train_mnist.reshape(10, 100, 28, 28, 1))
            y_tensor = tf.convert_to_tensor(y_train_mnist.reshape(10, 100, 10))
            # tmp = x_train_mnist.shape[0] / default_batch_size
            # x_tensor = tf.convert_to_tensor(x_train_mnist.reshape(tmp, default_batch_size, 28, 28, 1))
            # y_tensor = tf.convert_to_tensor(y_train_mnist.reshape(tmp, default_batch_size, 10))
            dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
            return dataset.make_initializable_iterator()

    if framework == "pytorch":
        # Create tensors from data
        x_train_tens = torch.from_numpy(x_train_mnist)
        x_train_tens = x_train_tens.float()
        y_train_tens = torch.from_numpy(y_train_mnist)
        dataset = torch.utils.data.TensorDataset(x_train_tens, y_train_tens)
        return DataLoader(dataset=dataset, batch_size=default_batch_size, shuffle=True)

    return None


@pytest.fixture
def image_data_generator(framework, is_tf_version_2, get_default_mnist_subset, image_iterator, default_batch_size):
    def _image_data_generator(**kwargs):
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        if framework == "keras":
            return KerasDataGenerator(
                iterator=image_iterator,
                size=x_train_mnist.shape[0],
                batch_size=default_batch_size,
            )

        if framework == "tensorflow":
            if not is_tf_version_2:
                return TensorFlowDataGenerator(
                    sess=kwargs["sess"], iterator=image_iterator, iterator_type="initializable", iterator_arg={},
                    size=x_train_mnist.shape[0],
                    batch_size=default_batch_size
                )

        if framework == "pytorch":
            return PyTorchDataGenerator(iterator=image_iterator, size=x_train_mnist.shape[0],
                                        batch_size=default_batch_size)

    return _image_data_generator


@pytest.fixture
def get_image_classifier_list(framework):
    def _get_image_classifier_list(one_classifier=False, **kwargs):
        sess = None
        if framework == "keras":
            classifier_list = [get_image_classifier_kr(**kwargs)]
        if framework == "tensorflow":
            classifier, sess = get_image_classifier_tf(**kwargs)
            classifier_list = [classifier]
        if framework == "pytorch":
            classifier_list = [get_image_classifier_pt()]
        if framework == "scikitlearn":
            logging.warning("{0} doesn't have an image classifier defined yet".format(framework))
            classifier_list = None

        if classifier_list is None:
            return None, None

        if one_classifier:
            return classifier_list[0], sess

        return classifier_list, sess

    return _get_image_classifier_list


@pytest.fixture
def get_tabular_classifier_list(framework):
    def _get_tabular_classifier_list(clipped=True):
        if framework == "keras":
            if clipped:
                classifier_list = [get_tabular_classifier_kr()]
            else:
                classifier = get_tabular_classifier_kr()
                classifier_list = [KerasClassifier(model=classifier.model, use_logits=False, channels_first=True)]

        if framework == "tensorflow":
            if clipped:
                classifier, _ = get_tabular_classifier_tf()
                classifier_list = [classifier]
            else:
                logging.warning("{0} doesn't have an uncliped classifier defined yet".format(framework))
                classifier_list = None

        if framework == "pytorch":
            if clipped:
                classifier_list = [get_tabular_classifier_pt()]
            else:
                logging.warning("{0} doesn't have an uncliped classifier defined yet".format(framework))
                classifier_list = None

        if framework == "scikitlearn":
            return get_tabular_classifier_scikit_list(clipped=False)

        return classifier_list

    return _get_tabular_classifier_list


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


@pytest.fixture(scope="session")
def framework(request):
    mlFramework = request.config.getoption("--mlFramework")
    if mlFramework not in art_supported_frameworks:
        raise Exception("mlFramework value {0} is unsupported. Please use one of these valid values: {1}".format(
            mlFramework, " ".join(art_supported_frameworks)))
    # if utils_test.is_valid_framework(mlFramework):
    #     raise Exception("The mlFramework specified was incorrect. Valid options available
    #     are {0}".format(art_supported_frameworks))
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
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris), _, _ = load_dataset('iris')

    yield (x_train_iris, y_train_iris), (x_test_iris, y_test_iris)


@pytest.fixture(scope="function")
def get_iris_dataset(load_iris_dataset, framework):
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
def get_default_mnist_subset(framework, get_mnist_dataset, default_dataset_subset_sizes):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train, n_test = default_dataset_subset_sizes

    if framework == "pytorch":
        x_train_mnist = np.reshape(x_train_mnist, (x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_mnist = np.reshape(x_test_mnist, (x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)

    yield (x_train_mnist[:n_train], y_train_mnist[:n_train]), (x_test_mnist[:n_test], y_test_mnist[:n_test])


@pytest.fixture(scope="session")
def load_mnist_dataset():
    logging.info("Loading mnist")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = load_dataset('mnist')
    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)


@pytest.fixture(scope="function")
def create_test_dir():
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture(scope="function")
def get_mnist_dataset(load_mnist_dataset, framework):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist_dataset

    if framework == "pytorch":
        x_train_mnist = np.reshape(x_train_mnist, (x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
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
def only_with_platform(request, framework):
    if request.node.get_closest_marker('only_with_platform'):
        if framework not in request.node.get_closest_marker('only_with_platform').args:
            pytest.skip('skipped on this platform: {}'.format(framework))


# ART test fixture to skip test for specific mlFramework values
# eg: @pytest.mark.skipMlFramework("tensorflow","scikitlearn")
@pytest.fixture(autouse=True)
def skip_by_platform(request, framework):
    if request.node.get_closest_marker('skipMlFramework'):
        if framework in request.node.get_closest_marker('skipMlFramework').args:
            pytest.skip('skipped on this platform: {}'.format(framework))


@pytest.fixture
def make_customer_record():
    def _make_customer_record(name):
        return {"name": name, "orders": []}

    return _make_customer_record


@pytest.fixture(autouse=True)
def framework_agnostic(request, framework):
    if request.node.get_closest_marker('framework_agnostic'):
        if framework is not default_framework:
            pytest.skip('framework agnostic test skipped for framework : {}'.format(framework))
