import json
import os
import warnings

from keras.models import model_from_json

from src.classifiers.cnn import CNN
from src.classifiers.resnet import ResNet
from src.layers.activations import BoundedReLU
from src.utils import make_directory

custom_objects = {'BoundedReLU': BoundedReLU}

def save_classifier(classifier, file_path="./model/"):
    """ Saves classifier in the given location

    :param classifier: model to save
    :param str file_path: path to file
    :param dict comp_param: optional compilation parameters
    :return: None
    """
    make_directory(file_path.rsplit('/', 1)[0])
    # save classifier params
    with open(os.path.join(file_path, 'params.json'), 'w') as fp:
        params = {"class_name": type(classifier).__name__, "defences": classifier.defences}
        json.dump(params, fp)

    # serialize model to JSON
    with open(os.path.join(file_path, "model.json"), "w") as json_file:
        model_json = classifier.model.to_json()
        json_file.write(model_json)

    # serialize weights to HDF5
    classifier.model.save_weights(os.path.join(file_path, "weights.h5"))

    # save compilation params to json
    if classifier.comp_param:
        with open(os.path.join(file_path, 'comp_par.json'), 'w') as fp:
            json.dump(classifier.comp_param, fp)

def load_classifier(file_path, weights_name="weights.h5"):
    """ Loads a classifier from given location and tries to compile it

    :param file_path: folder to load the model from (full path)
    :param weights_name: name of the file containing the weights
    :return: Classifier
    """

    # load json and create model
    with open(os.path.join(file_path, "model.json"), "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects=custom_objects)

    # load params to decide what classifier to create

    with open(os.path.join(file_path, "params.json"), "r") as json_file:
        params_json = json.load(json_file)

    if "defences" in params_json.keys():
        defences = params_json["defences"]
    else:
        defences = None

    classifier = globals()[params_json["class_name"]](model=model, defences=defences)

    # load weights into new model
    classifier.model.load_weights(os.path.join(file_path, weights_name))

    # try to load comp param and compile model
    try:
        with open(os.path.join(file_path, 'comp_par.json'), 'r') as fp:
            classifier.comp_par = json.load(fp)
            classifier.model.compile(**classifier.comp_par)
    except OSError:
        warnings.warn("Compilation parameters not found. The loaded model will need to be compiled.")

    return classifier