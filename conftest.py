# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import importlib
import json
import logging
import os
import shutil
import tempfile
from typing import Dict, List, TYPE_CHECKING, Union
import warnings

import numpy as np
import pytest
import requests

from art.data_generators import (
    KerasDataGenerator,
    MXDataGenerator,
    PyTorchDataGenerator,
    TensorFlowDataGenerator,
    TensorFlowV2DataGenerator,
)
from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.estimators.classification import KerasClassifier
from tests.utils import (
    ARTTestFixtureNotImplemented,
    get_attack_classifier_pt,
    get_image_classifier_kr,
    get_image_classifier_kr_functional,
    get_image_classifier_kr_tf,
    get_image_classifier_kr_tf_functional,
    get_image_classifier_kr_tf_with_wildcard,
    get_image_classifier_mxnet_custom_ini,
    get_image_classifier_pt,
    get_image_classifier_pt_functional,
    get_image_classifier_tf,
    get_image_classifier_hf,
    get_image_gan_tf_v2,
    get_image_generator_tf_v2,
    get_tabular_classifier_kr,
    get_tabular_classifier_pt,
    get_tabular_classifier_scikit_list,
    get_tabular_classifier_tf,
    load_dataset,
    master_seed,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

deep_learning_frameworks = [
    "keras", "tensorflow1", "tensorflow2", "tensorflow2v1", "pytorch", "kerastf", "mxnet", "jax", "huggingface",
]
non_deep_learning_frameworks = ["scikitlearn"]

art_supported_frameworks = []
art_supported_frameworks.extend(deep_learning_frameworks)
art_supported_frameworks.extend(non_deep_learning_frameworks)

master_seed(1234)


def get_default_framework():
    import tensorflow as tf

    if tf.__version__[0] == "2":
        default_framework = "tensorflow2"
    else:
        default_framework = "tensorflow1"

    return default_framework


def pytest_addoption(parser):
    parser.addoption(
        "--framework",
        action="store",
        default=get_default_framework(),
        help="ART tests allow you to specify which framework to use. The default framework used is `tensorflow`. "
        "Other options available are {0}".format(art_supported_frameworks),
    )


@pytest.fixture
def image_dl_estimator_defended(framework):
    def _image_dl_estimator_defended(one_classifier=False, **kwargs):
        sess = None
        classifier = None

        clip_values = (0, 1)
        fs = FeatureSqueezing(bit_depth=2, clip_values=clip_values)

        defenses = []
        if kwargs.get("defenses") is None:
            defenses.append(fs)
        else:
            if "FeatureSqueezing" in kwargs.get("defenses"):
                defenses.append(fs)
            if "JpegCompression" in kwargs.get("defenses"):
                defenses.append(JpegCompression(clip_values=clip_values, apply_predict=True))
            if "SpatialSmoothing" in kwargs.get("defenses"):
                defenses.append(SpatialSmoothing())
            del kwargs["defenses"]

        if framework == "tensorflow2":
            classifier, _ = get_image_classifier_tf(**kwargs)

        if framework == "keras":
            classifier = get_image_classifier_kr(**kwargs)

        if framework == "kerastf":
            classifier = get_image_classifier_kr_tf(**kwargs)

        if framework == "pytorch":
            classifier = get_image_classifier_pt(**kwargs)
            for i, defense in enumerate(defenses):
                if "channels_first" in defense.params:
                    defenses[i].channels_first = classifier.channels_first

        if classifier is not None:
            classifier.set_params(preprocessing_defences=defenses)
        else:
            raise ARTTestFixtureNotImplemented(
                "no defended image estimator", image_dl_estimator_defended.__name__, framework, {"defenses": defenses}
            )

        return classifier, sess

    return _image_dl_estimator_defended


@pytest.fixture(scope="function")
def image_dl_estimator_for_attack(framework, image_dl_estimator, image_dl_estimator_defended):
    def _image_dl_estimator_for_attack(attack, defended=False, **kwargs):
        if defended:
            potential_classifier, _ = image_dl_estimator_defended(**kwargs)
        else:
            potential_classifier, _ = image_dl_estimator(**kwargs)
        image_dl_estimator_for_attack
        classifier_list = [potential_classifier]
        classifier_tested = [
            potential_classifier
            for potential_classifier in classifier_list
            if all(t in type(potential_classifier).__mro__ for t in attack._estimator_requirements)
        ]

        if len(classifier_tested) == 0:
            raise ARTTestFixtureNotImplemented(
                "no estimator available", image_dl_estimator_for_attack.__name__, framework, {"attack": attack}
            )
        return classifier_tested[0]

    return _image_dl_estimator_for_attack


@pytest.fixture
def estimator_for_attack(framework):
    # TODO DO NOT USE THIS FIXTURE this needs to be refactored into image_dl_estimator_for_attack
    def _get_attack_classifier_list(**kwargs):
        if framework == "pytorch":
            return get_attack_classifier_pt(**kwargs)

        raise ARTTestFixtureNotImplemented("no estimator available", image_dl_estimator_for_attack.__name__, framework)

    return _get_attack_classifier_list


@pytest.fixture(autouse=True)
def setup_tear_down_framework(framework):
    # Ran before each test
    if framework == "tensorflow1" or framework == "tensorflow2":
        import tensorflow as tf

        if tf.__version__[0] != "2":
            tf.reset_default_graph()

    if framework == "tensorflow2v1":
        import tensorflow.compat.v1 as tf1

        tf1.reset_default_graph()
    yield True

    # Ran after each test
    if framework == "keras":
        import keras

        keras.backend.clear_session()


@pytest.fixture
def image_iterator(framework, get_default_mnist_subset, default_batch_size):
    (x_train_mnist, y_train_mnist), (_, _) = get_default_mnist_subset

    def _get_image_iterator():
        if framework == "keras" or framework == "kerastf":
            from keras.preprocessing.image import ImageDataGenerator

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

        if framework in ["tensorflow1", "tensorflow2v1"]:
            import tensorflow.compat.v1 as tf

            x_tensor = tf.convert_to_tensor(x_train_mnist.reshape(10, 100, 28, 28, 1))
            y_tensor = tf.convert_to_tensor(y_train_mnist.reshape(10, 100, 10))
            dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
            return dataset.make_initializable_iterator()

        if framework == "tensorflow2":
            import tensorflow as tf

            dataset = tf.data.Dataset.from_tensor_slices((x_train_mnist, y_train_mnist)).batch(default_batch_size)
            return dataset

        if framework in ["pytorch", "huggingface"]:
            import torch

            # Create tensors from data
            x_train_tens = torch.from_numpy(x_train_mnist)
            x_train_tens = x_train_tens.float()
            y_train_tens = torch.from_numpy(y_train_mnist)
            dataset = torch.utils.data.TensorDataset(x_train_tens, y_train_tens)
            return torch.utils.data.DataLoader(dataset=dataset, batch_size=default_batch_size, shuffle=True)

        if framework == "mxnet":
            from mxnet import gluon

            dataset = gluon.data.dataset.ArrayDataset(x_train_mnist, y_train_mnist)
            return gluon.data.DataLoader(dataset, batch_size=5, shuffle=True)

        return None

    return _get_image_iterator


@pytest.fixture
def image_data_generator(framework, get_default_mnist_subset, image_iterator, default_batch_size):
    def _image_data_generator(**kwargs):
        (x_train_mnist, y_train_mnist), (_, _) = get_default_mnist_subset
        image_it = image_iterator()
        data_generator = None

        if framework == "keras" or framework == "kerastf":
            data_generator = KerasDataGenerator(
                iterator=image_it,
                size=x_train_mnist.shape[0],
                batch_size=default_batch_size,
            )

        if framework in ["tensorflow1", "tensorflow2v1"]:
            data_generator = TensorFlowDataGenerator(
                sess=kwargs["sess"],
                iterator=image_it,
                iterator_type="initializable",
                iterator_arg={},
                size=x_train_mnist.shape[0],
                batch_size=default_batch_size,
            )

        if framework == "tensorflow2":
            data_generator = TensorFlowV2DataGenerator(
                iterator=image_it,
                size=x_train_mnist.shape[0],
                batch_size=default_batch_size,
            )

        if framework in ["pytorch", "huggingface"]:
            data_generator = PyTorchDataGenerator(
                iterator=image_it, size=x_train_mnist.shape[0], batch_size=default_batch_size
            )

        if framework == "mxnet":
            data_generator = MXDataGenerator(
                iterator=image_it, size=x_train_mnist.shape[0], batch_size=default_batch_size
            )

        return data_generator

    return _image_data_generator


@pytest.fixture
def store_expected_values(request):
    """
    Stores expected values to be retrieved by the expected_values fixture
    Note1: Numpy arrays MUST be converted to list before being stored as json
    Note2: It's possible to store both a framework independent and framework specific value. If both are stored the
    framework specific value will be used
    :param request:
    :return:
    """

    def _store_expected_values(values_to_store, framework=""):

        framework_name = framework
        if framework_name:
            framework_name = "_" + framework_name

        file_name = request.node.location[0].split("/")[-1][:-3] + ".json"

        try:
            with open(
                os.path.join(os.path.dirname(__file__), os.path.dirname(request.node.location[0]), file_name), "r"
            ) as f:
                expected_values = json.load(f)
        except FileNotFoundError:
            expected_values = {}

        test_name = request.node.name + framework_name
        expected_values[test_name] = values_to_store

        with open(
                os.path.join(os.path.dirname(__file__), os.path.dirname(request.node.location[0]), file_name), "w"
        ) as f:
            json.dump(expected_values, f, indent=4)

    return _store_expected_values


@pytest.fixture
def expected_values(framework, request):
    """
    Retrieves the expected values that were stored using the store_expected_values fixture
    :param request:
    :return:
    """

    file_name = request.node.location[0].split("/")[-1][:-3] + ".json"

    framework_name = framework
    if framework_name:
        framework_name = "_" + framework_name

    def _expected_values():
        with open(
            os.path.join(os.path.dirname(__file__), os.path.dirname(request.node.location[0]), file_name), "r"
        ) as f:
            expected_values = json.load(f)

            # searching first for any framework specific expected value
            framework_specific_values = request.node.name + framework_name
            if framework_specific_values in expected_values:
                return expected_values[framework_specific_values]
            elif request.node.name in expected_values:
                return expected_values[request.node.name]
            else:
                raise ARTTestFixtureNotImplemented(
                    "Couldn't find any expected values for test {0}".format(request.node.name),
                    expected_values.__name__,
                    framework_name,
                )

    return _expected_values


@pytest.fixture(scope="session")
def get_image_classifier_mx_model():
    import mxnet  # lgtm [py/import-and-import-from]

    # TODO needs to be made parameterizable once Mxnet allows multiple identical models to be created in one session
    from_logits = True

    class Model(mxnet.gluon.nn.Block):
        def __init__(self, **kwargs):
            super(Model, self).__init__(**kwargs)
            self.model = mxnet.gluon.nn.Sequential()
            self.model.add(
                mxnet.gluon.nn.Conv2D(
                    channels=1,
                    kernel_size=7,
                    activation="relu",
                ),
                mxnet.gluon.nn.MaxPool2D(pool_size=4, strides=4),
                mxnet.gluon.nn.Flatten(),
                mxnet.gluon.nn.Dense(
                    10,
                    activation=None,
                ),
            )

        def forward(self, x):
            y = self.model(x)
            if from_logits:
                return y

            return y.softmax()

    model = Model()
    custom_init = get_image_classifier_mxnet_custom_ini()
    model.initialize(init=custom_init)
    return model


@pytest.fixture
def get_image_classifier_mx_instance(get_image_classifier_mx_model, mnist_shape):
    import mxnet  # lgtm [py/import-and-import-from]
    from art.estimators.classification import MXClassifier

    model = get_image_classifier_mx_model

    def _get_image_classifier_mx_instance(from_logits=True):
        if from_logits is False:
            # due to the fact that only 1 instance of get_image_classifier_mx_model can be created in one session
            # this will be resolved once Mxnet allows for 2 models with identical weights to be created in 1 session
            raise ARTTestFixtureNotImplemented(
                "Currently only supporting Mxnet classifier with from_logit set to True",
                get_image_classifier_mx_instance.__name__,
                framework,
            )

        loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(from_logits=from_logits)
        trainer = mxnet.gluon.Trainer(model.collect_params(), "sgd", {"learning_rate": 0.1})

        # Get classifier
        mxc = MXClassifier(
            model=model,
            loss=loss,
            input_shape=mnist_shape,
            # input_shape=(28, 28, 1),
            nb_classes=10,
            optimizer=trainer,
            ctx=None,
            channels_first=True,
            clip_values=(0, 1),
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=(0.0, 1.0),
        )

        return mxc

    return _get_image_classifier_mx_instance


@pytest.fixture
def supported_losses_types(framework):
    def supported_losses_types():
        if framework == "keras":
            return ["label", "function_losses", "function_backend"]
        if framework == "kerastf":
            # if loss_type is not "label" and loss_name not in ["categorical_hinge", "kullback_leibler_divergence"]:
            return ["label", "function", "class"]

        raise ARTTestFixtureNotImplemented(
            "Could not find supported_losses_types", supported_losses_types.__name__, framework
        )

    return supported_losses_types


@pytest.fixture
def supported_losses_logit(framework):
    def _supported_losses_logit():
        if framework == "keras":
            return ["categorical_crossentropy_function_backend", "sparse_categorical_crossentropy_function_backend"]
        if framework == "kerastf":
            # if loss_type is not "label" and loss_name not in ["categorical_hinge", "kullback_leibler_divergence"]:
            return [
                "categorical_crossentropy_function",
                "categorical_crossentropy_class",
                "sparse_categorical_crossentropy_function",
                "sparse_categorical_crossentropy_class",
            ]
        raise ARTTestFixtureNotImplemented(
            "Could not find supported_losses_logit", supported_losses_logit.__name__, framework
        )

    return _supported_losses_logit


@pytest.fixture
def supported_losses_proba(framework):
    def _supported_losses_proba():
        if framework == "keras":
            return [
                "categorical_hinge_function_losses",
                "categorical_crossentropy_label",
                "categorical_crossentropy_function_losses",
                "categorical_crossentropy_function_backend",
                "sparse_categorical_crossentropy_label",
                "sparse_categorical_crossentropy_function_losses",
                "sparse_categorical_crossentropy_function_backend",
            ]
        if framework == "kerastf":
            return [
                "categorical_hinge_function",
                "categorical_hinge_class",
                "categorical_crossentropy_label",
                "categorical_crossentropy_function",
                "categorical_crossentropy_class",
                "sparse_categorical_crossentropy_label",
                "sparse_categorical_crossentropy_function",
                "sparse_categorical_crossentropy_class",
                # "kullback_leibler_divergence_function",
                "kullback_leibler_divergence_class",
            ]

        raise ARTTestFixtureNotImplemented(
            "Could not find supported_losses_proba", supported_losses_proba.__name__, framework
        )

    return _supported_losses_proba


@pytest.fixture
def image_dl_generator(framework):
    def _image_dl_generator(**kwargs):
        if framework == "tensorflow2":
            return get_image_generator_tf_v2(64, 100)
        raise ARTTestFixtureNotImplemented("no test generator available", image_dl_generator.__name__, framework)

    return _image_dl_generator


@pytest.fixture
def image_dl_gan(framework):
    sess = None

    def _image_dl_gan(**kwargs):
        if framework == "tensorflow2":
            return get_image_gan_tf_v2(**kwargs), sess
        raise ARTTestFixtureNotImplemented("no test gan available", image_dl_gan.__name__, framework)

    return _image_dl_gan


@pytest.fixture
def image_dl_estimator(framework, get_image_classifier_mx_instance):
    def _image_dl_estimator(functional=False, **kwargs):
        sess = None
        wildcard = False
        classifier = None

        if kwargs.get("wildcard") is not None:
            if kwargs.get("wildcard") is True:
                wildcard = True
            del kwargs["wildcard"]

        if framework == "keras":
            if wildcard is False and functional is False:
                if functional:
                    classifier = get_image_classifier_kr_functional(**kwargs)
                else:
                    try:
                        classifier = get_image_classifier_kr(**kwargs)
                    except NotImplementedError:
                        raise ARTTestFixtureNotImplemented(
                            "This combination of loss function options is currently not supported.",
                            image_dl_estimator.__name__,
                            framework,
                        )
        if framework in ["tensorflow1", "tensorflow2", "tensorflow2v1"]:
            if wildcard is False and functional is False:
                classifier, sess = get_image_classifier_tf(**kwargs, framework=framework)
                return classifier, sess
        if framework == "pytorch":
            if not wildcard:
                if functional:
                    classifier = get_image_classifier_pt_functional(**kwargs)
                else:
                    classifier = get_image_classifier_pt(**kwargs)
        if framework == "kerastf":
            if wildcard:
                classifier = get_image_classifier_kr_tf_with_wildcard(**kwargs)
            else:
                if functional:
                    classifier = get_image_classifier_kr_tf_functional(**kwargs)
                else:
                    classifier = get_image_classifier_kr_tf(**kwargs)

        if framework == "mxnet":
            if wildcard is False and functional is False:
                classifier = get_image_classifier_mx_instance(**kwargs)

        if framework == "huggingface":
            if not wildcard:
                classifier = get_image_classifier_hf(**kwargs)

        if classifier is None:
            raise ARTTestFixtureNotImplemented(
                "no test deep learning estimator available", image_dl_estimator.__name__, framework
            )

        return classifier, sess

    return _image_dl_estimator


@pytest.fixture
def art_warning(request):
    def _art_warning(exception):
        if type(exception) is ARTTestFixtureNotImplemented:
            if request.node.get_closest_marker("framework_agnostic"):
                if not request.node.get_closest_marker("parametrize"):
                    raise Exception(
                        "This test has marker framework_agnostic decorator which means it will only be ran "
                        "once. However the ART test exception was thrown, hence it is never run fully. "
                    )
            elif (
                request.node.get_closest_marker("only_with_platform")
                and len(request.node.get_closest_marker("only_with_platform").args) == 1
            ):
                raise Exception(
                    "This test has marker only_with_platform decorator which means it will only be ran "
                    "once. However the ARTTestFixtureNotImplemented exception was thrown, hence it is "
                    "never run fully. "
                )

            # NotImplementedErrors are raised in ART whenever a test model does not exist for a specific
            # model/framework combination. By catching there here, we can provide a report at the end of each
            # pytest run list all models requiring to be implemented.
            warnings.warn(UserWarning(exception))
        else:
            raise exception

    return _art_warning


@pytest.fixture
def decision_tree_estimator(framework):
    def _decision_tree_estimator(clipped=True):
        if framework == "scikitlearn":
            return get_tabular_classifier_scikit_list(clipped=clipped, model_list_names=["decisionTreeClassifier"])[0]

        raise ARTTestFixtureNotImplemented(
            "no test decision_tree_classifier available", decision_tree_estimator.__name__, framework
        )

    return _decision_tree_estimator


@pytest.fixture
def tabular_dl_estimator(framework):
    def _tabular_dl_estimator(clipped=True):
        classifier = None
        if framework == "keras":
            if clipped:
                classifier = get_tabular_classifier_kr()
            else:
                kr_classifier = get_tabular_classifier_kr()
                classifier = KerasClassifier(model=kr_classifier.model, use_logits=False, channels_first=True)

        if framework == "tensorflow1" or framework == "tensorflow2":
            if clipped:
                classifier, _ = get_tabular_classifier_tf()

        if framework == "pytorch":
            if clipped:
                classifier = get_tabular_classifier_pt()

        if classifier is None:
            raise ARTTestFixtureNotImplemented(
                "no deep learning tabular estimator available", tabular_dl_estimator.__name__, framework
            )
        return classifier

    return _tabular_dl_estimator


@pytest.fixture(scope="function")
def create_test_image(create_test_dir):
    test_dir = create_test_dir
    # Download one ImageNet pic for tests
    url = "http://farm1.static.flickr.com/163/381342603_81db58bea4.jpg"
    result = requests.get(url, stream=True)
    if result.status_code == 200:
        image = result.raw.read()
        f = open(os.path.join(test_dir, "test.jpg"), "wb")
        f.write(image)
        f.close()

    yield os.path.join(test_dir, "test.jpg")


@pytest.fixture(scope="session")
def framework(request):
    ml_framework = request.config.getoption("--framework")
    if ml_framework == "tensorflow":
        import tensorflow as tf

        if tf.__version__[0] == "2":
            ml_framework = "tensorflow2"
        else:
            ml_framework = "tensorflow1"

    if ml_framework not in art_supported_frameworks:
        raise Exception(
            "framework value {0} is unsupported. Please use one of these valid values: {1}".format(
                ml_framework, " ".join(art_supported_frameworks)
            )
        )
    # if utils_test.is_valid_framework(framework):
    #     raise Exception("The framework specified was incorrect. Valid options available
    #     are {0}".format(art_supported_frameworks))
    return ml_framework


@pytest.fixture(scope="session")
def default_batch_size():
    yield 16


@pytest.fixture(scope="session")
def load_iris_dataset():
    logging.info("Loading Iris dataset")
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris), _, _ = load_dataset("iris")

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
def load_diabetes_dataset():
    logging.info("Loading Diabetes dataset")
    (x_train_diabetes, y_train_diabetes), (x_test_diabetes, y_test_diabetes), _, _ = load_dataset("diabetes")

    yield (x_train_diabetes, y_train_diabetes), (x_test_diabetes, y_test_diabetes)


@pytest.fixture(scope="function")
def get_diabetes_dataset(load_diabetes_dataset, framework):
    (x_train_diabetes, y_train_diabetes), (x_test_diabetes, y_test_diabetes) = load_diabetes_dataset

    x_train_diabetes_original = x_train_diabetes.copy()
    y_train_diabetes_original = y_train_diabetes.copy()
    x_test_diabetes_original = x_test_diabetes.copy()
    y_test_diabetes_original = y_test_diabetes.copy()

    yield (x_train_diabetes, y_train_diabetes), (x_test_diabetes, y_test_diabetes)

    np.testing.assert_array_almost_equal(x_train_diabetes_original, x_train_diabetes, decimal=3)
    np.testing.assert_array_almost_equal(y_train_diabetes_original, y_train_diabetes, decimal=3)
    np.testing.assert_array_almost_equal(x_test_diabetes_original, x_test_diabetes, decimal=3)
    np.testing.assert_array_almost_equal(y_test_diabetes_original, y_test_diabetes, decimal=3)


@pytest.fixture(scope="session")
def default_dataset_subset_sizes():
    n_train = 1000
    n_test = 100
    yield n_train, n_test


@pytest.fixture()
def mnist_shape(framework):
    if framework in ["pytorch", "mxnet", "huggingface"]:
        return (1, 28, 28)
    else:
        return (28, 28, 1)


@pytest.fixture()
def cifar10_shape(framework):
    if framework == "pytorch" or framework == "mxnet":
        return (3, 32, 32)
    else:
        return (32, 32, 3)


@pytest.fixture()
def get_default_mnist_subset(get_mnist_dataset, default_dataset_subset_sizes, mnist_shape):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train, n_test = default_dataset_subset_sizes

    x_train_mnist = np.reshape(x_train_mnist, (x_train_mnist.shape[0],) + mnist_shape).astype(np.float32)
    x_test_mnist = np.reshape(x_test_mnist, (x_test_mnist.shape[0],) + mnist_shape).astype(np.float32)

    yield (x_train_mnist[:n_train], y_train_mnist[:n_train]), (x_test_mnist[:n_test], y_test_mnist[:n_test])


@pytest.fixture()
def get_default_cifar10_subset(get_cifar10_dataset, default_dataset_subset_sizes, cifar10_shape):
    (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = get_cifar10_dataset
    n_train, n_test = default_dataset_subset_sizes

    x_train_cifar10 = np.reshape(x_train_cifar10, (x_train_cifar10.shape[0],) + cifar10_shape).astype(np.float32)
    x_test_cifar10 = np.reshape(x_test_cifar10, (x_test_cifar10.shape[0],) + cifar10_shape).astype(np.float32)

    yield (x_train_cifar10[:n_train], y_train_cifar10[:n_train]), (x_test_cifar10[:n_test], y_test_cifar10[:n_test])


@pytest.fixture(scope="session")
def load_mnist_dataset():
    logging.info("Loading mnist")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = load_dataset("mnist")
    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)


@pytest.fixture(scope="session")
def load_cifar10_dataset():
    logging.info("Loading cifar10")
    (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10), _, _ = load_dataset("cifar10")
    yield (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10)


@pytest.fixture(scope="function")
def create_test_dir():
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture(scope="function")
def get_mnist_dataset(load_mnist_dataset, mnist_shape):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist_dataset

    x_train_mnist = np.reshape(x_train_mnist, (x_train_mnist.shape[0],) + mnist_shape).astype(np.float32)
    x_test_mnist = np.reshape(x_test_mnist, (x_test_mnist.shape[0],) + mnist_shape).astype(np.float32)

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


@pytest.fixture(scope="function")
def get_cifar10_dataset(load_cifar10_dataset, cifar10_shape):
    (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = load_cifar10_dataset

    x_train_cifar10 = np.reshape(x_train_cifar10, (x_train_cifar10.shape[0],) + cifar10_shape).astype(np.float32)
    x_test_cifar10 = np.reshape(x_test_cifar10, (x_test_cifar10.shape[0],) + cifar10_shape).astype(np.float32)

    x_train_cifar10_original = x_train_cifar10.copy()
    y_train_cifar10_original = y_train_cifar10.copy()
    x_test_cifar10_original = x_test_cifar10.copy()
    y_test_cifar10_original = y_test_cifar10.copy()

    yield (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10)

    # Check that the test data has not been modified, only catches changes in attack.generate if self has been used
    np.testing.assert_array_almost_equal(x_train_cifar10_original, x_train_cifar10, decimal=3)
    np.testing.assert_array_almost_equal(y_train_cifar10_original, y_train_cifar10, decimal=3)
    np.testing.assert_array_almost_equal(x_test_cifar10_original, x_test_cifar10, decimal=3)
    np.testing.assert_array_almost_equal(y_test_cifar10_original, y_test_cifar10, decimal=3)


# ART test fixture to skip test for specific framework values
# eg: @pytest.mark.only_with_platform("tensorflow")
@pytest.fixture(autouse=True)
def only_with_platform(request, framework):
    if request.node.get_closest_marker("only_with_platform"):
        if framework not in request.node.get_closest_marker("only_with_platform").args:
            pytest.skip("skipped on this platform: {}".format(framework))


# ART test fixture to skip test for specific framework values
# eg: @pytest.mark.skip_framework("tensorflow", "keras", "pytorch", "scikitlearn",
# "mxnet", "kerastf", "non_dl_frameworks", "dl_frameworks")
@pytest.fixture(autouse=True)
def skip_by_framework(request, framework):
    if request.node.get_closest_marker("skip_framework"):
        framework_to_skip_list = list(request.node.get_closest_marker("skip_framework").args)
        if "dl_frameworks" in framework_to_skip_list:
            framework_to_skip_list.extend(deep_learning_frameworks)

        if "non_dl_frameworks" in framework_to_skip_list:
            framework_to_skip_list.extend(non_deep_learning_frameworks)

        if "tensorflow" in framework_to_skip_list:
            framework_to_skip_list.append("tensorflow1")
            framework_to_skip_list.append("tensorflow2")
            framework_to_skip_list.append("tensorflow2v1")

        if framework in framework_to_skip_list:
            pytest.skip("skipped on this platform: {}".format(framework))


@pytest.fixture
def make_customer_record():
    def _make_customer_record(name):
        return {"name": name, "orders": []}

    return _make_customer_record


@pytest.fixture(autouse=True)
def framework_agnostic(request, framework):
    if request.node.get_closest_marker("framework_agnostic"):
        if framework != get_default_framework():
            pytest.skip("framework agnostic test skipped for framework : {}".format(framework))


# ART test fixture to skip test for specific required modules
# eg: @pytest.mark.skip_module("deepspeech_pytorch", "apex.amp", "object_detection")
@pytest.fixture(autouse=True)
def skip_by_module(request):
    if request.node.get_closest_marker("skip_module"):
        modules_from_args = request.node.get_closest_marker("skip_module").args

        # separate possible parent modules and test them first
        modules_parents = [m.split(".", 1)[0] for m in modules_from_args]

        # merge with modules including submodules (Note: sort ensures that parent comes first)
        modules_full = sorted(set(modules_parents).union(modules_from_args))

        for module in modules_full:
            if module in modules_full:
                module_spec = importlib.util.find_spec(module)
                module_found = module_spec is not None

                if not module_found:
                    pytest.skip(f"Test skipped because package {module} not available.")


@pytest.fixture()
def fix_get_rcnn():

    from art.estimators.estimator import BaseEstimator, LossGradientsMixin
    from art.estimators.object_detection.object_detector import ObjectDetectorMixin

    class DummyObjectDetector(ObjectDetectorMixin, LossGradientsMixin, BaseEstimator):
        def __init__(self):
            self._clip_values = (0, 1)
            self.channels_first = False
            self._input_shape = None
            self._compute_loss_count = 1

        def loss_gradient(self, x: np.ndarray, y: None, **kwargs):
            return np.ones_like(x)

        def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs):
            raise NotImplementedError

        def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
            dict_i = {"boxes": np.array([[0.1, 0.2, 0.3, 0.4]]), "labels": np.array([[2]]), "scores": np.array([[0.8]])}
            return [dict_i] * x.shape[0]

        @property
        def native_label_is_pytorch_format(self):
            return True

        @property
        def input_shape(self):
            return self._input_shape

        def compute_losses(
            self, x: np.ndarray, y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]]
        ) -> Dict[str, np.ndarray]:

            losses_dict = {
                "loss_classifier": np.array(0.43572357, dtype=float),
                "loss_box_reg": np.array(0.17341757, dtype=float),
                "loss_objectness": np.array(0.02198849, dtype=float),
                "loss_rpn_box_reg": np.array(0.03471708, dtype=float),
            }

            return losses_dict

        def compute_loss(
            self, x: np.ndarray, y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]], **kwargs
        ) -> Union[np.ndarray, "torch.Tensor"]:
            self._compute_loss_count += 1
            loss = 0.43572357 / self._compute_loss_count
            return loss

    frcnn = DummyObjectDetector()
    return frcnn


@pytest.fixture()
def fix_get_goturn():

    from art.estimators.estimator import BaseEstimator, LossGradientsMixin
    from art.estimators.object_tracking.object_tracker import ObjectTrackerMixin

    class DummyObjectTracker(ObjectTrackerMixin, LossGradientsMixin, BaseEstimator):
        def __init__(self):
            super().__init__(
                model=None,
                clip_values=(0, 1),
                preprocessing_defences=None,
                postprocessing_defences=None,
                preprocessing=(0, 1),
            )

            import torch

            self.channels_first = False
            self._input_shape = None
            self.postprocessing_defences = None
            self.device = torch.device("cpu")

        def loss_gradient(self, x: np.ndarray, y: None, **kwargs):
            return np.ones_like(x)

        def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs):
            raise NotImplementedError

        def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
            boxes_list = list()
            for i in range(x.shape[1]):
                boxes_list.append([0.1, 0.2, 0.3, 0.4])

            dict_i = {"boxes": np.array(boxes_list), "labels": np.array([[2]]), "scores": np.array([[0.8]])}
            return [dict_i] * x.shape[0]

        @property
        def native_label_is_pytorch_format(self):
            return True

        @property
        def input_shape(self):
            return self._input_shape

    goturn = DummyObjectTracker()
    return goturn
