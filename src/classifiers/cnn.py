from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization

from src.classifiers.utils import activation

def cnn_model(input_shape, act='relu', bnorm=False, logits=False, input_ph=None, nb_filters=64, nb_classes=10, act_params={}):
    """Returns a ConvolutionalNeuralNetwork model using Keras sequential model
    
    :param tuple input_shape: shape of the input images
    :param str act: type of the intermediate activation functions
    :param bool bnorm: whether to apply batch normalization after each layer or not
    :param bool logits: If set to False, returns a Keras model, otherwise will also
                return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                (needed if returning logits)
                ("ph" stands for placeholder but it need not actually be a
                placeholder)
    :param int nb_filters: number of convolutional filters per layer
    :param int nb_classes: the number of output classes
    :param dict act_params: dict of params for activation layers
    :return: CNN model
    :rtype: keras.model
    """

    model = Sequential()

    layers = [Conv2D(nb_filters, (8, 8), strides=(2, 2), padding="same", input_shape=input_shape),
              Conv2D((nb_filters * 2), (6, 6), strides=(2, 2), padding="valid"),
              Conv2D((nb_filters * 2), (5, 5), strides=(1, 1), padding="valid"),
              Flatten()]

    for layer in layers:
        model.add(layer)
        model.add(activation(act, **act_params))
        if bnorm:
            model.add(BatchNormalization())
    model.add(Dense(nb_classes))

    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model