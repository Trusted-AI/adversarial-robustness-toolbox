import json
import os
import warnings

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as k
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from src.classifiers.classifier import Classifier
from src.layers.activations import BoundedReLU
from src.defences.preprocessing import feature_squeezing
custom_objects = {'BoundedReLU': BoundedReLU}


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
    """Bayesian convolutional neural network implementation using Keras sequential model"""
    def __init__(self, input_shape=None, include_end=True, act='relu', bnorm=False, input_ph=None, nb_filters=64,
                 nb_classes=10, nb_mc_samples=100, act_params={}, model=None, defences=None, preproc=None,
                 dataset="mnist"):
        """Instantiates a Bayesian convolutional neural network model using Keras sequential model
        
        :param input_shape: (tuple) shape of the input images
        :param include_end: (boolean, default True) whether to include a softmax layer at the end or not
        :param act: (string) type of the intermediate activation functions
        :param bnorm: (boolean, default False) whether to apply batch normalization after each layer or not
        :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
        :param nb_filters: (positive int) number of convolutional filters per layer
        :param nb_classes: (positive int, default 10) the number of output classes
        :param nb_mc_samples: (positive int, default 100) the number of Monte Carlo samples to be used while predicting
        :param act_params: (dict) dict of params for activation layers
        """
        if model is None:
            model = Sequential(name='bnn')

            if "mnist" in dataset:
                layers = mnist_layers(input_shape, nb_filters)
            elif "cifar10" in dataset:
                layers = cifar10_layers(input_shape, nb_filters)
            elif "stl10" in dataset:
                raise NotImplementedError("No BNN architecture is defined for dataset '{0}'.".format(dataset))

            # Check if at least one of the layers is Dropout
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

        self.nb_mc_samples = nb_mc_samples
        super(BNN, self).__init__(model, defences, preproc)

    def _mc_preds(self, x, batch_size=10):
        """Compute predictions for Monte Carlo samples

        :param x: input data
        :param batch_size: (positive int, default 10) batch sizez
        :return: list of predictions
        """
        # Define a numpy generator for x
        datagen = ImageDataGenerator()
        x_gen = datagen.flow(x, None, batch_size=batch_size, shuffle=False)

        mc_output = k.function([self.model.layers[0].input, k.learning_phase()], [self.model.layers[-1].output])
        learning_phase = True  # use dropout at test time

        samples_seen = 0
        mc_out = []

        while samples_seen < x.shape[0]:
            x_ = x_gen.__next__()
            mc_samples = [mc_output([x_, learning_phase])[0] for _ in range(self.nb_mc_samples)]
            mc_samples = np.array(mc_samples)  # [#samples x batch size x #classes]
            
            if samples_seen == 0:
                mc_out = mc_samples
            else: 
                mc_out = np.concatenate((mc_out, mc_samples), axis=1)
            samples_seen += x_.shape[0]
            
        return mc_out

    def predict(self, x_val, **kwargs):
        if self.feature_squeeze:
            x = feature_squeezing(x_val, self.bit_depth)
        else:
            x = x_val

        x = self._preprocess(x)        
        mc_samples = self._mc_preds(x, **kwargs)
        
        return np.mean(mc_samples, axis=0)

    def evaluate(self, x_val, y_val, **kwargs):
        # TODO include other metrics in evaluate and perform evaluate with _mc_preds
        if self.feature_squeeze:
            x = feature_squeezing(x_val, self.bit_depth)
        else:
            x = x_val

        x = self._preprocess(x)
        return self.model.evaluate(x, y_val, **kwargs)

    @staticmethod
    def _load_from_bnn(file_path, weights_name="weights.h5", nb_mc_samples=None):
        """Load a classifier from file_path and try to compile it

        :param file_path: folder to load the model from (full path)
        :param weights_name: name of the file containing the weights
        :param nb_mc_samples: number of Monte Carlo samples to be used while predicting
        :return: Classifier
        """
        # Load json and create model
        with open(os.path.join(file_path, "model.json"), "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects=custom_objects)

        # Load params to decide what classifier to create
        with open(os.path.join(file_path, "params.json"), "r") as json_file:
            params_json = json.load(json_file)

        if "defences" in params_json.keys():
            defences = params_json["defences"]
        else:
            defences = None

        if nb_mc_samples is not None:
            classifier = BNN(model=model, defences=defences, nb_mc_samples=nb_mc_samples)
        else:
            classifier = BNN(model=model, defences=defences)

        # Load weights into new model
        classifier.model.load_weights(os.path.join(file_path, weights_name))

        # Try to load compilation parameters and compile model
        try:
            with open(os.path.join(file_path, 'comp_par.json'), 'r') as fp:
                classifier.comp_par = json.load(fp)
                if classifier.comp_par["optimizer"] == "sgd":
                    classifier.comp_par["optimizer"] = SGD(lr=1e-4, momentum=0.9)
                classifier.model.compile(**classifier.comp_par)
        except OSError:
            warnings.warn("Compilation parameters not found. The loaded model will need to be compiled.")

        return classifier
