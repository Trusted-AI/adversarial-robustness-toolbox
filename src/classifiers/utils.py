from __future__ import absolute_import, division, print_function

import json
import os
import warnings

from keras.models import model_from_json
from keras.optimizers import SGD

from src.layers.activations import BoundedReLU
from src.utils import make_directory

custom_objects = {'BoundedReLU': BoundedReLU}


def save_classifier(classifier, file_path="./model/"):
    """Saves classifier in the given location

    :param classifier: Model to save
    :param str file_path: Path to file
    """
    make_directory(file_path.rsplit('/', 1)[0])
    # Save classifier params
    with open(os.path.join(file_path, 'params.json'), 'w') as fp:
        params = {"class_name": type(classifier).__name__, "defences": classifier.defences}
        json.dump(params, fp)

    # Serialize model to JSON
    with open(os.path.join(file_path, "model.json"), "w") as json_file:
        model_json = classifier.model.to_json()
        json_file.write(model_json)

    # Serialize weights to HDF5
    classifier.model.save_weights(os.path.join(file_path, "weights.h5"))

    # Save compilation params to json
    if classifier.comp_param:
        with open(os.path.join(file_path, 'comp_par.json'), 'w') as fp:
            try:
                json.dump(classifier.comp_param, fp)
            except:
                fp.seek(0)
                json.dump({"loss": 'categorical_crossentropy', "optimizer": "sgd",
                           "metrics": ['accuracy']}, fp)
                fp.truncate()


def load_classifier(file_path, weights_name="weights.h5"):
    """Loads a classifier from given location and tries to compile it

    :param file_path: folder to load the model from (full path)
    :param weights_name: name of the file containing the weights
    :return: Classifier
    """
    from src.classifiers.bnn import BNN
    from src.classifiers.cnn import CNN
    from src.classifiers.mlp import MLP
    from src.classifiers.resnet import ResNet

    # Load json and create model
    with open(os.path.join(file_path, "model.json"), "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects=custom_objects)

    # Load params to decide what classifier to create
    with open(os.path.join(file_path, "params.json"), "r") as json_file:
        params_json = json.load(json_file)

    if "defences" in params_json.keys():
        defences = params_json["defences"]
    else:
        defences = None

    classifier = globals()[params_json["class_name"]](model=model, defences=defences)

    # Load weights into new model
    classifier.model.load_weights(os.path.join(file_path, weights_name))

    # Try to load compilation parameters and compile model
    try:
        with open(os.path.join(file_path, 'comp_par.json'), 'r') as fp:
            classifier.comp_par = json.load(fp)
            if classifier.comp_par["optimizer"] == "sgd":
                classifier.comp_par["optimizer"] = SGD(lr=1e-4, momentum=0.9)
            classifier.model.compile(**classifier.comp_par)
    except OSError:
        warnings.warn("Compilation parameters not found. The loaded model will need to be compiled.")

    return classifier


def check_is_fitted(model, parameters, error_msg=None):
    """
    Checks if the model is fitted by asserting the presence of the fitted parameters in the model.

    :param model: The model instance
    :param parameters: The name of the parameter or list of names of parameters that are fitted by the model
    :param error_msg: (string) Custom error message to be printed if the model is not fitted. Default message is
    'This model is not fitted yet. Call 'fit' with appropriate arguments before using this method.'
    :return: (bool) True if the model is fitted
    :raises: TypeError
    """
    if error_msg is None:
        error_msg = "This model is not fitted yet. Call 'fit' with appropriate arguments before using this method."

    if not hasattr(model, 'fit'):
        raise TypeError("%s cannot be fitted." % model)

    if not isinstance(parameters, (list, tuple)):
        parameters = [parameters]

    if not all([hasattr(model, param) for param in parameters]):
        return False

    return True
