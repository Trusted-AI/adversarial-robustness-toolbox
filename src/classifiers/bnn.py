from keras.constraints import maxnorm
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from src.classifiers.classifier import Classifier
from keras.preprocessing.image import ImageDataGenerator

def mnist_layers(input_shape, nb_filters):

    layers = [Conv2D(nb_filters, (8, 8), strides=(2, 2), padding="same", input_shape=input_shape),
              "activation",
              Conv2D((nb_filters * 2), (6, 6), strides=(2, 2), padding="valid"),
              "activation",
              Conv2D((nb_filters * 2), (5, 5), strides=(1, 1), padding="valid"),
              "activation",
              Dropout(0.5),
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

class BNN(Classifier):

    def __init__(self, input_shape=None, include_end=True, act='relu', bnorm=False, input_ph=None, nb_filters=64,
                 nb_classes=10,nb_MC_samples=100,act_params={}, model=None, defences=None, preproc=None, dataset="mnist"):

        """Instantiates a Bayesian ConvolutionalNeuralNetwork model using Keras sequential model
        
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
        :param int nb_MC_samples: the number of MC samples to be used while predicting
        :param dict act_params: dict of params for activation layers
        :param str 
        :rtype: keras.model
        """

        if model is None:

            model = Sequential(name='cnn')

            if "mnist" in dataset:
                layers = mnist_layers(input_shape, nb_filters)

            elif "cifar10" in dataset:
                layers = cifar10_layers(input_shape, nb_filters)

            ## check if atleast one of them is Dropout
            assert ('Dropout' in [layer.__class__.__name__ for layer in layers])
            
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

        self.nb_MC_samples = nb_MC_samples

        super(BNN, self).__init__(model, defences, preproc)

    def _mc_preds(self, x, batch_size = 10, **kwargs):


        #define a numpy generator for x
        datagen = ImageDataGenerator()
        x_gen = datagen.flow(x,None,batch_size=batch_size,shuffle=False)

        MC_output = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        learning_phase = True  # use dropout at test time

        samples_seen = 0
        MC_out = []

        while samples_seen < x.shape[0]:
            x_ = x_gen.__next__()
            
            MC_samples = [MC_output([x_, learning_phase])[0] for _ in range(self.nb_MC_samples)]
            MC_samples = np.array(MC_samples)  # [#samples x batch size x #classes]
            
            if samples_seen == 0:
                MC_out = MC_samples
            else: 
                MC_out = np.concatenate((MC_out,MC_samples),axis=1)
            samples_seen+=x_.shape[0]
            
        return MC_out


    def predict(self, x_val, **kwargs):
        
        if self.feature_squeeze:
            x = feature_squeezing(x_val, self.bit_depth)

        else:
            x = x_val

        x = self._preprocess(x)        
        mc_samples = self._mc_preds(x,**kwargs)
        
        return np.mean(mc_samples,axis=0)

    def evaluate(self, x_val, y_val, **kwargs):
        #TODO: include other metrics in evaluate and perform evaluate with _mc_preds

        if self.feature_squeeze:
            x = feature_squeezing(x_val, self.bit_depth)

        else:
            x = x_val

        x = self._preprocess(x)
        return self.model.evaluate(x, y_val, **kwargs)
