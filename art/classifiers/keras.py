from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from art.classifiers import Classifier


class KerasClassifier(Classifier):
    """
    The supported backends for Keras are TensorFlow and Theano.
    """
    def __init__(self, clip_values, model, use_logits=False, defences=None):
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: Keras model
        :type model: `keras.models.Sequential`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        """
        import keras.backend as k

        # TODO Generalize loss function?
        super(KerasClassifier, self).__init__(clip_values, defences)

        self._model = model
        self._input = model.input
        self._output = model.output
        _, self._nb_classes = k.int_shape(model.output)
        self._input_shape = k.int_shape(model.input)[1:]

        # Get predictions and loss function
        label_ph = k.placeholder(shape=(None,))
        if not use_logits:
            if k.backend() == 'tensorflow':
                preds, = self._output.op.inputs
                loss = k.sparse_categorical_crossentropy(label_ph, preds, from_logits=True)
            else:
                loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)

                # Convert predictions to logits for consistency with the other cases
                eps = 10e-8
                preds = k.log(k.clip(self._output, eps, 1. - eps))
        else:
            preds = self._output
            loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)
        loss_grads = k.gradients(loss, self._input)

        if k.backend() == 'tensorflow':
            loss_grads = loss_grads[0]
        elif k.backend() == 'cntk':
            raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

        # Compute gradient per class, with and without the softmax activation
        class_grads_logits = [k.gradients(preds[:, i], self._input)[0] for i in range(self._nb_classes)]
        class_grads = [k.gradients(k.softmax(preds)[:, i], self._input)[0] for i in range(self._nb_classes)]

        # Set loss, grads and prediction functions
        self._loss = k.function([self._input], [loss])
        self._loss_grads = k.function([self._input, label_ph], [loss_grads])
        self._class_grads_logits = k.function([self._input], class_grads_logits)
        self._class_grads = k.function([self._input], class_grads)
        self._preds = k.function([self._input], [preds])

    def loss_gradient(self, inputs, labels):
        """
        Compute the gradient of the loss function w.r.t. `inputs`.

        :param inputs: Sample input with shape as expected by the model.
        :type inputs: `np.ndarray`
        :param labels: Correct labels, one-vs-rest encoding.
        :type labels: `np.ndarray`
        :return: Array of gradients of the same shape as the inputs.
        :rtype: `np.ndarray`
        """
        return self._loss_grads([inputs, np.argmax(labels, axis=1)])[0]

    def class_gradient(self, inputs, logits=False):
        """
        Compute per-class derivatives w.r.t. `input`.

        :param inputs: Sample input with shape as expected by the model.
        :type inputs: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)`.
        :rtype: `np.ndarray`
        """
        if logits:
            return np.swapaxes(np.array(self._class_grads_logits([inputs])), 0, 1)
        else:
            return np.swapaxes(np.array(self._class_grads([inputs])), 0, 1)

    def predict(self, inputs, logits=False):
        """
        Perform prediction for a batch of inputs.

        :param inputs: Test set.
        :type inputs: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        # Apply defences
        inputs = self._apply_defences_predict(inputs)

        preds = self._preds([inputs])[0]
        if not logits:
            exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            preds = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def fit(self, inputs, outputs, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(inputs, outputs)`.

        :param inputs: Training data.
        :type inputs: `np.ndarray`
        :param outputs: Labels.
        :type outputs: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        import keras.backend as k
        k.set_learning_phase(1)

        # Apply defences
        inputs, outputs = self._apply_defences_fit(inputs, outputs)

        gen = generator(inputs, outputs, batch_size)
        self._model.fit_generator(gen, steps_per_epoch=inputs.shape[0] / batch_size, epochs=nb_epochs)


def generator(data, labels, batch_size=128):
    """
    Minimal data generator for batching large datasets.

    :param data: The data sample to batch.
    :type data: `np.ndarray`
    :param labels: The labels for `data`. The first dimension has to match the first dimension of `data`.
    :type labels: `np.ndarray`
    :param batch_size: The size of the batches to produce.
    :type batch_size: `int`
    :return: A batch of size `batch_size` of random samples from `(data, labels)`
    :rtype: `tuple(np.ndarray, np.ndarray)`
    """
    while True:
        indices = np.random.randint(data.shape[0], size=batch_size)
        yield data[indices], labels[indices]
