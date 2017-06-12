import json
import os
import warnings

from keras.layers import Activation
from keras.models import model_from_json

from src.layers.activations import BoundedReLU
from src.utils import make_directory

custom_objects = {'BoundedReLU' :BoundedReLU}

def activation(act, **kwargs):
    """ Creates and returns the Layer object corresponding to `act` activation function

    :param str act: name of the activation function
    :return: 
    :rtype: keras.Layer
    """
    if act in ['relu']:
        return Activation(act)
    elif act == 'brelu':
        return BoundedReLU(**kwargs)
    else:
        raise Exception("Activation function not supported.")

def save_model(model, filepath="./model/", comp_param=None):
    """ Saves model in the given location

    :param model: model to save
    :param str filepath: filepath
    :param dict comp_param: optional compilation parameters
    :return: None
    """
    make_directory(filepath.rsplit('/', 1)[0])
    # serialize model to JSON
    model_json = model.to_json()
    with open(filepath + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath + "weights.h5")
    # save compilation params to json
    if comp_param:
        with open(filepath + 'comp_par.json', 'w') as fp:
            json.dump(comp_param, fp)


def load_model(filepath, weightsname="weights.h5"):
    """ Loads a model from given location and tries to compile it

    :param filepath: folder to load the model from (full path)
    :param weightsname: name of the file containing the weights
    :return: keras sequential model 
    """
    # load json and create model
    with open(os.path.join(filepath, "model.json"), "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json, custom_objects=custom_objects)
    # load weights into new model
    model.load_weights(os.path.join(filepath, weightsname))
    # try to load comp param and compile model
    try:
        with open(os.path.join(filepath, 'comp_par.json'), 'r') as fp:
            comp_par = json.load(fp)
            model.compile(**comp_par)
    except OSError:
        warnings.warn("Compilation parameters not found. The loaded model will need to be compiled.")

    return model