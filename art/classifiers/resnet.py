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

# -*- coding: utf-8 -*-
# Code adapted from https://github.com/fchollet/deep-learning-models
from __future__ import absolute_import, division, print_function

import keras.backend as k
from keras import layers
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model

from art.classifiers.classifier import Classifier


def identity_block(input_tensor, kernel_size, filters, stage, block, bnorm):
    """The identity block is the block that has no conv layer at shortcut

    :param input_tensor: input tensor
    :param kernel_size: (default 3) the kernel size of middle conv layer at main path
    :param filters: list of integers, the filterss of 3 conv layer at main path
    :param stage: integer, current stage label, used for generating layer names
    :param block: 'a','b'..., current block label, used for generating layer names
    :param bnorm: (boolean) True for batch normalization
    :return: Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if k.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)

    if bnorm:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

    if bnorm:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    if bnorm:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


class ResNet(Classifier):
    """Instantiates a ResNet model using Keras sequential model."""
    def __init__(self, input_shape=None, include_end=True, act='relu', bnorm=False, input_ph=None, nb_filters=64,
                 nb_classes=10, act_params={}, model=None, defences=None, preproc=None, dataset='mnist'):
        """Instantiates a ResNet model using Keras sequential model

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
        if model is None:
            if 'mnist' not in dataset:
                raise NotImplementedError("No ResNet architecture is defined for dataset '{0}'.".format(dataset))

            img_input = Input(shape=input_shape)
            if k.image_data_format() == 'channels_last':
                bn_axis = 3
            else:
                bn_axis = 1

            x = Conv2D(nb_filters, (8, 8), strides=(2, 2), padding="same", input_shape=input_shape, name="conv1")(
                img_input)

            if bnorm:
                x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)

            x = Activation('relu')(x)
            x = Conv2D(nb_filters * 2, (6, 6), strides=(2, 2), padding="valid", name="conv2")(x)

            if bnorm:
                x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x)

            x = Activation('relu')(x)
            x = identity_block(x, 3, [nb_filters, nb_filters, nb_filters * 2], stage=2, block='a', bnorm=bnorm)
            x = MaxPooling2D((3, 3), name='max_pool')(x)
            x = Flatten()(x)
            x = Dense(nb_classes)(x)

            if include_end:
                x = Activation('softmax')(x)

            # Create model
            model = Model(img_input, x, name='resnet')

        super(ResNet, self).__init__(model, defences, preproc)
