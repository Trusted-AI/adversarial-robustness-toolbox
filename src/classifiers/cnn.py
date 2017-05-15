import json
import warnings

from keras.models import Sequential,model_from_json
from keras.layers import Dense,Activation,Flatten,Conv2D

from src.layers.activations import BoundedReLU
from src.utils import make_directory

custom_objects = {'BoundedReLU':BoundedReLU}

def activation(act,**kwargs):
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

def cnn_model(input_shape, act='relu', logits=False, input_ph=None, nb_filters=64, nb_classes=10, act_params={}):
    """Returns a ConvolutionalNeuralNetwork model using Keras sequential model
    
    :param tuple input_shape: shape of the input images
    :param str act: type of the intermediate activation functions
    :param bool logits: If set to False, returns a Keras model, otherwise will also
                return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                (needed if returning logits)
                ("ph" stands for placeholder but it need not actually be a
                placeholder)
    :param int nb_filters: number of convolutional filters per layer
    :param int nb_classes: the number of output classes
    :return: CNN model
    :rtype: keras.model
    """

    model = Sequential()

    layers = [Conv2D(nb_filters,(8, 8),strides=(2, 2),padding="same",input_shape=input_shape),
              activation(act,**act_params),
              Conv2D((nb_filters * 2),(6, 6),strides=(2, 2),padding="valid"),
              activation(act,**act_params),
              Conv2D((nb_filters * 2),(5, 5),strides=(1, 1),padding="valid"),
              activation(act,**act_params),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

def save_model(model,filepath="./model/",comp_param=None):
    """ Saves model in the given location
    
    :param model: model to save
    :param str filepath: filepath
    :param dict comp_param: optional compilation parameters
    :return: None
    """
    make_directory(filepath.rsplit('/', 1)[0])
    # serialize model to JSON
    model_json = model.to_json()
    with open(filepath+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath+"weights.h5")
    # save compilation params to json
    if comp_param:
        with open(filepath+'comp_par.json', 'w') as fp:
            json.dump(comp_param,fp)

def load_model(filepath,weightsname="weights.h5"):
    """ Loads a model from given location and tries to compile it
    
    :param filepath: file to load the model from
    :return: keras sequential model 
    """
    # load json and create model
    with open(filepath + "model.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json,custom_objects=custom_objects)
    # load weights into new model
    model.load_weights(filepath + weightsname)
    # try to load comp param and compile model
    try:
        with open(filepath+'comp_par.json', 'r') as fp:
            comp_par = json.load(fp)
            model.compile(**comp_par)
    except OSError:
        warnings.warn("Compilation parameters not found. The loaded model will need to be compiled.")

    return model
