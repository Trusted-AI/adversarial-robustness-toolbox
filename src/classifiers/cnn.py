import config
from cleverhans.utils import conv_2d

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from layers.activations import BoundedReLU


def activation(act):
    if act in ['relu']:
        return Activation(act)
    elif act == 'brelu':
        return BoundedReLU()
    else:
        raise Exception("Activation function not supported.")

def cnn_model(input_shape, act='relu', logits=False, input_ph=None, nb_filters=64, nb_classes=10):
    """
    Returns a CNN model using Keras sequential model
    
    # Arguments
        input_shape (tuple): shape of the input images
        act (string): type of the intermediate activation functions
        logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
        input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
        nb_filters: number of convolutional filters per layer
        nb_classes: the number of output classes
    # Return
        CNN model
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
