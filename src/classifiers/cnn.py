import config
from cleverhans.utils import conv_2d

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from layers.activations import BoundedReLU


def activation(act):
    """ Creates and returns the Layer object corresponding to `act` activation function
    
    :param str act: name of the activation function
    :return: 
    :rtype: keras.Layer
    """
    if act in ['relu']:
        return Activation(act)
    elif act == 'brelu':
        return BoundedReLU()
    else:
        raise Exception("Activation function not supported.")

def cnn_model(input_shape, act='relu', logits=False, input_ph=None, nb_filters=64, nb_classes=10):
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

    layers = [conv_2d(nb_filters, (8, 8), (2, 2), "same", input_shape=input_shape),
              activation(act),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              activation(act),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              activation(act),
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
