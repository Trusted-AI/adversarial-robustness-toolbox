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

from keras.layers import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from art.classifiers.classifier import Classifier


def mnist_layers(input_shape):
    """
    MNIST layers for MLP
    :param input_shape: tuple, shape of the input without batch size
    :return: list of Keras layers
    """
    layers = [Flatten(input_shape=input_shape),
              Dense(400),
              "activation",
              Dense(400),
              "activation"]
    return layers


def cifar10_layers(input_shape):
    """
    CIFAR10 layers for MLP
    :param input_shape: tuple, shape of the input without batch size
    :return: list of Keras layers
    """
    # TODO implement layers for CIFAR10
    raise NotImplementedError("To be implemented")


class MLP(Classifier):
    """
    Implementation of the standard multi-layer perceptron
    """
    def __init__(self, input_shape=None, include_end=True, act='relu', bnorm=False, input_ph=None, nb_classes=10,
                 act_params={}, model=None, defences=None, preproc=None, dataset="mnist"):
        """Instantiate an MLP using Keras Sequential model

        :param input_shape: tuple, shape of the input images
        :param include_end: bool, True if a softmax layer is to be included at the end
        :param act: string, activation function
        :param bnorm: whether to apply batch normalization after each layer or not
        :param input_ph: The TensorFlow tensor for the input (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a placeholder)
        :param nb_classes: int, number of output classes
        :param act_params: dict of params for activation layers
        :param model: keras.model, None if it is to be created
        :param defences: string, defences to be used
        :param preproc: string, preprocessing to apply
        :param dataset: string, dataset to be used, default is MNIST
        :return: Keras.model object
        """
        if model is None:
            model = Sequential(name='mlp')
            layers = []
            if "mnist" in dataset:
                layers = mnist_layers(input_shape)
            elif "cifar10" in dataset:
                layers = cifar10_layers(input_shape)
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

        super(MLP, self).__init__(model, defences, preproc)
