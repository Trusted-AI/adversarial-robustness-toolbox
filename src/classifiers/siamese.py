""" Code adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py"""

from keras.models import Model
from keras.layers import Input, Lambda
from keras import backend as K

from src.classifiers.cnn import cnn_model

def get_base_network(netname, **kwargs):
    return cnn_model(**kwargs)

def rbf(inputs, radius):
    x, y = inputs
    return K.exp(-K.sum(K.square(x - y), axis=1, keepdims=True) / (2*radius))

def output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(1 - y_pred, 0)))

def siamese_model(input_shape, netname="cnn", act='relu', bnorm=False, input_ph=None, nb_filters=64, nb_classes=10,
                  act_params={}, radius = 1.):


    # network definition
    model = get_base_network(netname, input_shape=input_shape, act=act, bnorm=bnorm, include_end=False,
                                    input_ph=input_ph, nb_filters=nb_filters, nb_classes=nb_classes,
                                    act_params=act_params)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = model(input_a)
    processed_b = model(input_b)

    distance = Lambda(rbf, output_shape=output_shape, arguments={"radius":1})([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    return model