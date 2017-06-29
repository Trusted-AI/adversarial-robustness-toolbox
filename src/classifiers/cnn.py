from keras.constraints import maxnorm
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization

from src.classifiers.classifier import Classifier

def mnist_layers(input_shape, nb_filters):

    layers = [Conv2D(nb_filters, (8, 8), strides=(2, 2), padding="same", input_shape=input_shape),
              "activation",
              Conv2D((nb_filters * 2), (6, 6), strides=(2, 2), padding="valid"),
              "activation",
              Conv2D((nb_filters * 2), (5, 5), strides=(1, 1), padding="valid"),
              "activation",
              Flatten()]

    return layers


def cifar10_layers(input_shape, nb_filters,):

    print("cifar")

    layers = [Conv2D(nb_filters, (3, 3), padding="same", input_shape=input_shape),
              "activation",
              Dropout(0.2),
              Conv2D(nb_filters, (3, 3), padding="valid"),
              "activation",
              MaxPooling2D(pool_size=(2, 2)),
              Conv2D(nb_filters*2, (3, 3), padding='valid'),
              "activation",
              Dropout(0.2),
              Conv2D(nb_filters*2, (3, 3), padding='valid'),
              "activation",
              Dropout(0.2),
              Conv2D(nb_filters*4, (3, 3), padding='valid'),
              "activation",
              Dropout(0.2),
              Conv2D(nb_filters*4, (3, 3), padding='valid'),
              "activation",
              MaxPooling2D(pool_size=(2, 2)),
              Flatten(),
              Dropout(0.2),
              Dense(nb_filters*32, kernel_constraint=maxnorm(3)),
              "activation",
              Dropout(0.2),
              Dense(nb_filters*16, kernel_constraint=maxnorm(3)),
              "activation",
              Dropout(0.2)]

    return layers

class CNN(Classifier):

    def __init__(self, input_shape=None, include_end=True, act='relu', bnorm=False, input_ph=None, nb_filters=64,
                 nb_classes=10, act_params={}, model=None, defences=None, dataset="mnist"):

        """Instantiates a ConvolutionalNeuralNetwork model using Keras sequential model
        
        :param tuple input_shape: shape of the input images
        :param bool include_end: whether to include a softmax layer at the end or not
        :param str act: type of the intermediate activation functions
        :param bool bnorm: whether to apply batch normalization after each layer or not
        :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
        :param int nb_filters: number of convolutional filters per layer
        :param int nb_classes: the number of output classes
        :param dict act_params: dict of params for activation layers
        :param str 
        :rtype: keras.model
        """

        super(CNN, self).__init__(defences)

        if model:
            self.model = model

        else:
            self.model = Sequential(name='cnn')

            if "mnist" in dataset:
                layers = mnist_layers(input_shape, nb_filters)

            elif "cifar10" in dataset:
                layers = cifar10_layers(input_shape, nb_filters)

            for layer in layers:

                if layer == "activation":
                    self.model.add(self.get_activation(act, **act_params))
                    if bnorm:
                        self.model.add(BatchNormalization())
                else:
                    self.model.add(layer)

            self.model.add(Dense(nb_classes))

            if include_end:
                self.model.add(Activation('softmax'))