# MIT License
#
# Copyright (C) IBM Corporation 2018
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
from __future__ import absolute_import, division, print_function

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization

from art.classifiers.classifier import Classifier


def mnist_layers(input_shape, nb_filters):

    layers = [Conv2D(nb_filters, (8, 8), strides=(2, 2), padding="same", input_shape=input_shape),
              "activation",
              Conv2D((nb_filters * 2), (6, 6), strides=(2, 2), padding="valid"),
              "activation",
              Conv2D((nb_filters * 2), (5, 5), strides=(1, 1), padding="valid"),
              "activation",
              Flatten()]

    return layers


def cifar10_layers(input_shape, nb_filters):

    layers = [Conv2D(nb_filters // 2, (3, 3), padding="same", input_shape=input_shape),
              "activation",
              MaxPooling2D(pool_size=(2, 2)),
              Dropout(0.5),
              Conv2D(nb_filters, (3, 3), padding="valid"),
              "activation",
              MaxPooling2D(pool_size=(2, 2)),
              Dropout(0.5),
              Flatten(),
              Dense(500),
              "activation",
              Dropout(0.5)]

    return layers


class CNN(Classifier):
    """
    Implementation of a convolutional neural network using Keras sequential model
    """
    def __init__(self, input_shape=None, include_end=True, act='relu', bnorm=False, input_ph=None, nb_filters=64,
                 nb_classes=10, act_params={}, model=None, defences=None, preproc=None, dataset="mnist"):

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
        :rtype: keras.model object
        """
        if model is None:
            model = Sequential(name='cnn')
            layers = []

            if "mnist" in dataset:
                layers = mnist_layers(input_shape, nb_filters)
            elif "cifar10" in dataset:
                layers = cifar10_layers(input_shape, nb_filters)
            elif "stl10" in dataset:
                raise NotImplementedError("No CNN architecture is defined for dataset '{0}'.".format(dataset))

            for layer in layers:

                if layer == "activation":
                    model.add(self.get_activation(act, **act_params))
                    if bnorm:
                        model.add(BatchNormalization())
                else:
                    model.add(layer)

            model.add(Dense(nb_classes))

            if include_end:
                model.add(Activation('softmax'))

        super(CNN, self).__init__(model, defences, preproc)
