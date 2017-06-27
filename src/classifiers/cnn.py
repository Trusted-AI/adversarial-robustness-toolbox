from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization

from src.classifiers.classifier import Classifier

class CNN(Classifier):

    def __init__(self, input_shape=None, include_end=True, act='relu', bnorm=False, input_ph=None, nb_filters=64,
                 nb_classes=10, act_params={}, model=None, defences=None):

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

            layers = [Conv2D(nb_filters, (8, 8), strides=(2, 2), padding="same", input_shape=input_shape),
                      Conv2D((nb_filters * 2), (6, 6), strides=(2, 2), padding="valid"),
                      Conv2D((nb_filters * 2), (5, 5), strides=(1, 1), padding="valid"),
                      Flatten()]

            for layer in layers:
                self.model.add(layer)
                self.model.add(self.get_activation(act, **act_params))
                if bnorm:
                    self.model.add(BatchNormalization())
            self.model.add(Dense(nb_classes))

            if include_end:
                self.model.add(Activation('softmax'))