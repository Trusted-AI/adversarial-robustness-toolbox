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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import six

from art.classifiers import Classifier

logger = logging.getLogger(__name__)


class KerasClassifier(Classifier):
    """
    Wrapper class for importing Keras models. The supported backends for Keras are TensorFlow and Theano.
    """
    def __init__(self, clip_values, model, use_logits=False, channel_index=3, defences=None, preprocessing=(0, 1),
                 input_layer=0, output_layer=0, custom_activation=False):
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: Keras model
        :type model: `keras.models.Model`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param input_layer: Which layer to consider as the Input when the model has multple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the Output when the model has multiple output layers.
        :type output_layer: `int`
        :param custom_activation: True if the model uses the last activation other than softmax and requires to use the
               output probability rather than the logits by attacks.
        :type custom_activation: `bool`
        """
        super(KerasClassifier, self).__init__(clip_values=clip_values, channel_index=channel_index, defences=defences,
                                              preprocessing=preprocessing)

        self._model = model
        self._input_layer = input_layer
        self._output_layer = output_layer

        self._initialize_params(model, use_logits, input_layer, output_layer, custom_activation)

    def _initialize_params(self, model, use_logits, input_layer, output_layer, custom_activation):
        """
        Initialize most parameters of the classifier. This is a convenience function called by `__init__` and
        `__setstate__` to avoid code duplication.

        :param model: Keras model
        :type model: `keras.models.Model`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param input_layer: Which layer to consider as the Input when the model has multple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the Output when the model has multiple output layers.
        :type output_layer: `int`
        :param custom_activation: True if the model uses the last activation other than softmax and requires to use the
               output probability rather than the logits by attacks.
        :type custom_activation: `bool`
        """
        import keras.backend as k

        if hasattr(model, 'inputs'):
            self._input_layer = input_layer
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input
            self._input_layer = 0

        if hasattr(model, 'outputs'):
            self._output = model.outputs[output_layer]
            self._output_layer = output_layer
        else:
            self._output = model.output
            self._output_layer = 0

        _, self._nb_classes = k.int_shape(self._output)
        self._input_shape = k.int_shape(self._input)[1:]
        self._custom_activation = custom_activation
        logger.debug('Inferred %i classes and %s as input shape for Keras classifier.', self.nb_classes,
                     str(self.input_shape))

        # Get predictions and loss function
        label_ph = k.placeholder(shape=self._output.shape)
        if not hasattr(self._model, 'loss'):
            logger.warning('Keras model has no loss set. Trying to use `k.sparse_categorical_crossentropy`.')
            loss_function = k.sparse_categorical_crossentropy
        else:
            if isinstance(self._model.loss, six.string_types):
                loss_function = getattr(k, self._model.loss)
            else:
                loss_function = getattr(k, self._model.loss.__name__)

        self._use_logits = use_logits
        if not use_logits:
            if k.backend() == 'tensorflow':
                if custom_activation:
                    preds = self._output
                    loss_ = loss_function(label_ph, preds, from_logits=False)
                else:
                    # We get a list of tensors that comprise the final "layer" -> take the last element
                    preds = self._output.op.inputs[-1]
                    loss_ = loss_function(label_ph, preds, from_logits=True)
            else:
                loss_ = loss_function(label_ph, self._output, from_logits=use_logits)

                # Convert predictions to logits for consistency with the other cases
                eps = 10e-8
                preds = k.log(k.clip(self._output, eps, 1. - eps))
        else:
            preds = self._output
            loss_ = loss_function(label_ph, self._output, from_logits=use_logits)
        if preds == self._input:  # recent Tensorflow version does not allow a model with an output same as the input.
            preds = k.identity(preds)
        loss_grads = k.gradients(loss_, self._input)

        if k.backend() == 'tensorflow':
            loss_grads = loss_grads[0]
        elif k.backend() == 'cntk':
            raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

        # Set loss, grads and prediction functions
        self._preds_op = preds
        self._loss = loss_
        self._loss_grads = k.function([self._input, label_ph], [loss_grads])
        self._preds = k.function([self._input], [preds])

        # Set check for the shape of y for loss functions that do not take labels in one-hot encoding
        self._reduce_labels = (hasattr(self._loss.op, 'inputs') and
                               not all(len(input_.shape) == len(self._loss.op.inputs[0].shape)
                                       for input_ in self._loss.op.inputs))

        # Get the internal layer
        self._layer_names = self._get_layers()

    def loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        x_preproc = self._apply_processing(x)
        x_defences, y_defences = self._apply_defences(x_preproc, y, fit=False)

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        if self._reduce_labels:
            y_defences = np.argmax(y_defences, axis=1)

        grads = self._loss_grads([x_defences, y_defences])[0]
        grads = self._apply_defences_gradient(x_preproc, grads)
        grads = self._apply_processing_gradient(grads)
        assert grads.shape == x_preproc.shape

        return grads

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        # Check value of label for computing gradients
        if not (label is None or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
                or (isinstance(label, np.ndarray) and len(label.shape) == 1 and (label < self.nb_classes).all()
                    and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % str(label))

        self._init_class_grads(label=label, logits=logits)

        x_preproc = self._apply_processing(x)
        x_defences, _ = self._apply_defences(x_preproc, None)

        if label is None:
            # Compute the gradients w.r.t. all classes
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits([x_defences])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads([x_defences])), 0, 1)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits_idx[label]([x_defences])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads_idx[label]([x_defences])), 0, 1)

            assert grads.shape == (x_defences.shape[0], 1) + self.input_shape

        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            if logits:
                grads = np.array([self._class_grads_logits_idx[l]([x_defences]) for l in unique_label])
            else:
                grads = np.array([self._class_grads_idx[l]([x_defences]) for l in unique_label])
            grads = np.swapaxes(np.squeeze(grads, axis=1), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = np.expand_dims(grads[np.arange(len(grads)), lst], axis=1)

        grads = self._apply_defences_gradient(x_preproc, grads)
        grads = self._apply_processing_gradient(grads)

        return grads

    def predict(self, x, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        from art import NUMPY_DTYPE

        # Apply defences
        x_preproc = self._apply_processing(x)
        x_preproc, _ = self._apply_defences(x_preproc, None, fit=False)

        # Run predictions with batching
        preds = np.zeros((x_preproc.shape[0], self.nb_classes), dtype=NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preproc.shape[0] / float(batch_size)))):
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preproc.shape[0])
            preds[begin:end] = self._preds([x_preproc[begin:end]])[0]

            if not logits and not self._custom_activation:
                exp = np.exp(preds[begin:end] - np.max(preds[begin:end], axis=1, keepdims=True))
                preds[begin:end] = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs or the number of steps per epoch as part of this argument will result in as error.
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply preprocessing
        x_preproc = self._apply_processing(x)

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        if self._reduce_labels:
            x_preproc, y_preproc = self._apply_defences(x_preproc, y, fit=True)
            y_preproc = np.argmax(y_preproc, axis=1)
        else:
            x_preproc, y_preproc = self._apply_defences(x_preproc, y, fit=True)

        gen = generator_fit(x_preproc, y_preproc, batch_size)
        self._model.fit_generator(gen, steps_per_epoch=x_preproc.shape[0] / batch_size, epochs=nb_epochs, **kwargs)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in Keras, it will.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs as part of this argument will result in as error.
        :type kwargs: `dict`
        :return: `None`
        """
        from art.data_generators import KerasDataGenerator

        # Try to use the generator as a Keras native generator, otherwise use it through the `DataGenerator` interface
        if isinstance(generator, KerasDataGenerator) and not hasattr(self, 'defences'):
            try:
                self._model.fit_generator(generator.generator, epochs=nb_epochs, **kwargs)
            except ValueError:
                logger.info('Unable to use data generator as Keras generator. Now treating as framework-independent.')
                super(KerasClassifier, self).fit_generator(generator, nb_epochs=nb_epochs, **kwargs)
        else:
            super(KerasClassifier, self).fit_generator(generator, nb_epochs=nb_epochs, **kwargs)

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        return self._layer_names

    def get_activations(self, x, layer, batch_size=128):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        from art import NUMPY_DTYPE

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_name = self._layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        layer_output = self._model.get_layer(layer_name).output
        output_func = k.function([self._input], [layer_output])

        # Apply preprocessing and defences
        if x.shape == self.input_shape:
            x_preproc = np.expand_dims(x, 0)
        else:
            x_preproc = x
        x_preproc = self._apply_processing(x_preproc)
        x_preproc, _ = self._apply_defences(x_preproc, None, fit=False)
        assert len(x_preproc.shape) == 4

        # Determine shape of expected output and prepare array
        output_shape = output_func([x_preproc[0][None, ...]])[0].shape
        activations = np.zeros((x_preproc.shape[0],) + output_shape[1:], dtype=NUMPY_DTYPE)

        # Get activations with batching
        for batch_index in range(int(np.ceil(x_preproc.shape[0] / float(batch_size)))):
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preproc.shape[0])
            activations[begin:end] = output_func([x_preproc[begin:end]])[0]

        return activations

    def _init_class_grads(self, label=None, logits=False):
        import keras.backend as k

        if len(self._output.shape) == 2:
            nb_outputs = self._output.shape[1]
        else:
            raise ValueError('Unexpected output shape for classification in Keras model.')

        if label is None:
            logger.debug('Computing class gradients for all %i classes.', self.nb_classes)
            if logits:
                if not hasattr(self, '_class_grads_logits'):
                    class_grads_logits = [k.gradients(self._preds_op[:, i], self._input)[0]
                                          for i in range(nb_outputs)]
                    self._class_grads_logits = k.function([self._input], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads'):
                    class_grads = [k.gradients(k.softmax(self._preds_op)[:, i], self._input)[0]
                                   for i in range(nb_outputs)]
                    self._class_grads = k.function([self._input], class_grads)

        else:
            if isinstance(label, int):
                unique_labels = [label]
                logger.debug('Computing class gradients for class %i.', label)
            else:
                unique_labels = np.unique(label)
                logger.debug('Computing class gradients for classes %s.', str(unique_labels))

            if logits:
                if not hasattr(self, '_class_grads_logits_idx'):
                    self._class_grads_logits_idx = [None for _ in range(nb_outputs)]

                for current_label in unique_labels:
                    if self._class_grads_logits_idx[current_label] is None:
                        class_grads_logits = [k.gradients(self._preds_op[:, current_label], self._input)[0]]
                        self._class_grads_logits_idx[current_label] = k.function([self._input], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads_idx'):
                    self._class_grads_idx = [None for _ in range(nb_outputs)]

                for current_label in unique_labels:
                    if self._class_grads_idx[current_label] is None:
                        class_grads = [k.gradients(k.softmax(self._preds_op)[:, current_label], self._input)[0]]
                        self._class_grads_idx[current_label] = k.function([self._input], class_grads)

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        from keras.engine.topology import InputLayer

        layer_names = [layer.name for layer in self._model.layers[:-1] if not isinstance(layer, InputLayer)]
        logger.info('Inferred %i hidden layers on Keras classifier.', len(layer_names))

        return layer_names

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        import keras.backend as k

        if isinstance(train, bool):
            self._learning_phase = train
            k.set_learning_phase(int(train))

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os

        if path is None:
            from art import DATA_PATH
            full_path = os.path.join(DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        self._model.save(str(full_path))
        logger.info('Model saved in path: %s.', full_path)

    def __getstate__(self):
        """
        Use to ensure `KerasClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """
        import time

        state = self.__dict__.copy()

        # Remove the unpicklable entries
        del state['_model']
        del state['_input']
        del state['_output']
        del state['_preds_op']
        del state['_loss']
        del state['_loss_grads']
        del state['_preds']
        del state['_layer_names']

        model_name = str(time.time()) + '.h5'
        state['model_name'] = model_name
        self.save(model_name)
        return state

    def __setstate__(self, state):
        """
        Use to ensure `KerasClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to Keras
        import os
        from art import DATA_PATH
        from keras.models import load_model

        full_path = os.path.join(DATA_PATH, state['model_name'])
        model = load_model(str(full_path))

        self._model = model
        self._initialize_params(model, state['_use_logits'], state['_input_layer'], state['_output_layer'],
                                state['_custom_activation'])

    def __repr__(self):
        repr_ = "%s(clip_values=%r, model=%r, use_logits=%r, channel_index=%r, defences=%r, preprocessing=%r, " \
                "input_layer=%r, output_layer=%r, custom_activation=%r)" \
                % (self.__module__ + '.' + self.__class__.__name__,
                   self.clip_values, self._model, self._use_logits, self.channel_index, self.defences,
                   self.preprocessing, self._input_layer, self._output_layer, self._custom_activation)

        return repr_


def generator_fit(x, y, batch_size=128):
    """
    Minimal data generator for randomly batching large datasets.

    :param x: The data sample to batch.
    :type x: `np.ndarray`
    :param y: The labels for `x`. The first dimension has to match the first dimension of `x`.
    :type y: `np.ndarray`
    :param batch_size: The size of the batches to produce.
    :type batch_size: `int`
    :return: A batch of size `batch_size` of random samples from `(x, y)`
    :rtype: `tuple(np.ndarray, np.ndarray)`
    """
    while True:
        indices = np.random.randint(x.shape[0], size=batch_size)
        yield x[indices], y[indices]
