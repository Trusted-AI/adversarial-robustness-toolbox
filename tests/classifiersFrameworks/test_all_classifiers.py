import logging
import numpy as np
from os import listdir, path
import tempfile
import warnings

from tests.classifiersFrameworks.utils import fw_agnostic_backend_test_nb_classes
from tests.classifiersFrameworks.utils import fw_agnostic_backend_test_input_shape
from tests.classifiersFrameworks.utils import fw_agnostic_backend_test_repr
from tests.classifiersFrameworks.utils import fw_agnostic_backend_test_layers

logger = logging.getLogger(__name__)


def test_layers(get_default_mnist_subset, framework, is_tf_version_2, get_image_classifier_list):
    try:
        fw_agnostic_backend_test_layers(framework, is_tf_version_2, get_default_mnist_subset, get_image_classifier_list)
    except NotImplementedError as e:
        warnings.warn(UserWarning(e))

# def test_predict(get_default_mnist_subset, get_image_classifier_list):
#     (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
#
#     classifier, _ = get_image_classifier_list(one_classifier=True)
#
#     if classifier is not None:
#         y_predicted = classifier.predict(x_test_mnist[0:1])
#         y_expected = np.asarray(
#             [
#                 [
#                     0.12109935,
#                     0.0498215,
#                     0.0993958,
#                     0.06410097,
#                     0.11366927,
#                     0.04645343,
#                     0.06419806,
#                     0.30685693,
#                     0.07616713,
#                     0.05823758,
#                 ]
#             ]
#         )
#         np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)
#
#
# def test_nb_classes(get_image_classifier_list):
#     fw_agnostic_backend_test_nb_classes(get_image_classifier_list)
#
#
# def test_input_shape(framework, get_image_classifier_list):
#     fw_agnostic_backend_test_input_shape(framework, get_image_classifier_list)
#
#
# def test_save(get_image_classifier_list):
#     try:
#         classifier, _ = get_image_classifier_list(one_classifier=True)
#         if classifier is not None:
#             t_file = tempfile.NamedTemporaryFile()
#             model_path = t_file.name
#             t_file.close()
#             filename = "model_to_save"
#             classifier.save(filename, path=model_path)
#
#             assert path.exists(model_path)
#
#             created_model = False
#
#             for file in listdir(model_path):
#                 if filename in file:
#                     created_model = True
#             assert created_model
#
#     except NotImplementedError as e:
#         warnings.warn(UserWarning(e))
#
#
# def test_repr(framework, is_tf_version_2, get_image_classifier_list):
#     try:
#         classifier, _ = get_image_classifier_list(one_classifier=True)
#         if classifier is not None:
#             fw_agnostic_backend_test_repr(framework, is_tf_version_2, classifier)
#
#     except NotImplementedError as e:
#         warnings.warn(UserWarning(e))
