from __future__ import absolute_import, division, print_function, unicode_literals

import keras.backend as k
import numpy as np

from src.classifiers.classifier import Classifier


class KerasClassifier(Classifier):
    """
    The supported backends for Keras are TensorFlow and Theano.
    """
    def __init__(self, clip_values, model, use_logits=False):
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param clip_values: (tuple) Input range of values in the form (min, max)
        :param model: (k.Sequential) Keras model
        :param use_logits: (optional bool, default True) True if the output of the model are the logits
        """
        # TODO Handle compilation?
        # TODO Generalize loss function?
        super(KerasClassifier, self).__init__(clip_values)

        self._model = model
        self._input = model.input
        self._output = model.output
        _, self._nb_classes = k.int_shape(model.output)

        # Get predictions and loss function
        label_ph = k.placeholder(shape=(1,))
        if not use_logits:
            if k.backend() == 'tensorflow':
                preds, = self._output.op.inputs
                loss = k.sparse_categorical_crossentropy(label_ph, preds, from_logits=True)
            else:
                loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)

                # Convert predictions to logits
                eps = 10e-8
                preds = k.log(k.clip(self._output, eps, 1. - eps))
        else:
            preds = self._output
            loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)
        loss = k.squeeze(loss, axis=0)
        loss_grads = k.gradients(loss, self._input)

        if k.backend() == 'tensorflow':
            loss_grads = loss_grads[0]
        elif k.backend() == 'cntk':
            raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

        class_grads = [k.gradients(self._output[:, i], self._input)[0] for i in range(self._nb_classes)]

        # Set loss, grads and prediction functions
        self._loss = k.function([self._input], [loss])
        self._loss_grads = k.function([self._input, label_ph], [loss_grads])
        self._class_grads = k.function([self._input], class_grads)
        self._preds = k.function([self._input], [preds])

    def loss_gradient(self, input, label):
        return self._loss_grads([input, label])

    def class_gradient(self, input):
        return np.array(self._class_grads([input]))

    def predict(self, inputs):
        k.set_learning_phase(0)
        return self._preds([inputs])[0]

    def fit(self, inputs, outputs, batch_size=128, nb_epochs=10):
        k.set_learning_phase(1)
        gen = generator(inputs, outputs, batch_size)
        self._model.fit_generator(gen, steps_per_epoch=inputs.shape[0] / batch_size, epochs=nb_epochs)


def generator(data, labels, batch_size=128):
    """
    Minimal data generator for batching large datasets.
    :param data: (np.ndarray)
    :param labels: (np.ndarray) The labels for `data`. The first dimension has to match the first dimension of `data`.
    :param batch_size: (optional int) The batch size.
    :return: A batch of size `batch_size` of random samples from `(data, labels)`
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    while True:
        indices = np.random.randint(data.shape[0], size=batch_size)
        yield data[indices], labels[indices]
