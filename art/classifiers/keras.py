from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import six

from art.classifiers.classifier import Classifier, ImageClassifier, TextClassifier

logger = logging.getLogger(__name__)


class KerasClassifier(Classifier):
    """
    Wrapper class for importing Keras models. The supported backends for Keras are TensorFlow and Theano.
    """
    def __init__(self, model, use_logits=False, input_layer=0, output_layer=0, custom_activation=False):
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param model: Keras model
        :type model: `keras.models.Model`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param input_layer: Which layer to consider as the input when the model has multiple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the output when the model has multiple output layers.
        :type output_layer: `int`
        :param custom_activation: True if the model uses the last activation other than softmax and requires to use the
               output probability rather than the logits by attacks.
        :type custom_activation: `bool`
        """
        import keras.backend as k

        super(KerasClassifier, self).__init__()

        self._model = model
        if hasattr(model, 'inputs'):
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input

        if hasattr(model, 'outputs'):
            self._output = model.outputs[output_layer]
        else:
            self._output = model.output

        _, self._nb_classes = k.int_shape(self._output)
        # Treat binary classification separately
        if self._nb_classes == 1:
            self._nb_classes = 2
        logger.debug('Inferred %i classes for Keras classifier.', self.nb_classes)

        self._custom_activation = custom_activation

        # Get predictions and loss function
        label_ph = k.placeholder(shape=self._output.shape)
        if not hasattr(self._model, 'loss'):
            logger.warning('Keras model has no loss set. Trying to use `sparse_categorical_crossentropy`.')
            loss_function = k.sparse_categorical_crossentropy
        else:
            if isinstance(self._model.loss, str):
                loss_function = getattr(k, self._model.loss)
            else:
                loss_function = getattr(k, self._model.loss.__name__)

        if not use_logits:
            if k.backend() == 'tensorflow':
                if custom_activation:
                    preds = self._output
                    loss_ = loss_function(label_ph, preds, from_logits=False)
                else:
                    preds, = self._output.op.inputs
                    loss_ = loss_function(label_ph, preds, from_logits=True)
            else:
                loss_ = loss_function(label_ph, self._output, from_logits=use_logits)

                # Convert predictions to logits for consistency with the other cases
                eps = 10e-8
                preds = k.log(k.clip(self._output, eps, 1. - eps))
        else:
            preds = self._output
            loss_ = loss_function(label_ph, self._output, from_logits=use_logits)
        loss_grads = k.gradients(loss_, self._input)

        if k.backend() == 'tensorflow':
            loss_grads = loss_grads[0]
        elif k.backend() == 'cntk':
            raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

        # Set loss, grads and prediction functions
        self._preds_op = preds
        self._loss_op = loss_
        self._label_op = label_ph
        self._loss = k.function([self._input], [loss_])
        self._preds = k.function([self._input], [preds])

        try:
            self._loss_grads = k.function([self._input, label_ph], [loss_grads])
        except TypeError:
            pass

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
        x_ = self._apply_processing(x)
        grads = self._loss_grads([x_, y])[0]
        grads = self._apply_processing_gradient(grads)

        return grads

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when the `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        # Check value of label for computing gradients
        if not (label is None or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
                or (type(label) is np.ndarray and len(label.shape) == 1 and (label < self.nb_classes).all()
                    and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % str(label))

        self._init_class_grads(label=label, logits=logits)

        x_ = self._apply_processing(x)

        if label is None:
            # Compute the gradients w.r.t. all classes
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits([x_])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads([x_])), 0, 1)

            grads = self._apply_processing_gradient(grads)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits_idx[label]([x_])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads_idx[label]([x_])), 0, 1)

            grads = self._apply_processing_gradient(grads)
            assert grads.shape == (x_.shape[0], 1) + self.input_shape

        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            if logits:
                grads = np.array([self._class_grads_logits_idx[l]([x_]) for l in unique_label])
            else:
                grads = np.array([self._class_grads_idx[l]([x_]) for l in unique_label])
            grads = np.swapaxes(np.squeeze(grads, axis=1), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = np.expand_dims(grads[np.arange(len(grads)), lst], axis=1)

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
        import keras.backend as k
        k.set_learning_phase(0)

        # Apply defences
        x_ = self._apply_processing(x)
        x_ = self._apply_defences_predict(x_)

        # Run predictions with batching
        preds = np.zeros((x_.shape[0], *self._output.shape[1:]))
        for b in range(int(np.ceil(x_.shape[0] / float(batch_size)))):
            begin, end = b * batch_size,  min((b + 1) * batch_size, x_.shape[0])
            preds[begin:end] = self._preds([x_[begin:end]])[0]

            if not logits and not self._custom_activation:
                exp = np.exp(preds[begin:end] - np.max(preds[begin:end], axis=1, keepdims=True))
                preds[begin:end] = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def fit(self, x, y, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        import keras.backend as k
        k.set_learning_phase(1)

        # Apply preprocessing and defences
        x_ = self._apply_processing(x)
        x_, y_ = self._apply_defences_fit(x_, y)

        gen = generator_fit(x_, y_, batch_size)
        self._model.fit_generator(gen, steps_per_epoch=x_.shape[0] / batch_size, epochs=nb_epochs)

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

    def get_activations(self, x, layer):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
        elif type(layer) is int:
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_name = self._layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        layer_output = self._model.get_layer(layer_name).output
        output_func = k.function([self._input], [layer_output])

        # Apply preprocessing and defences
        x_ = self._apply_processing(x)
        x_ = self._apply_defences_predict(x_)

        return output_func([x_])[0]

    def _gen_init_class_grads(self, input_tensor, label=None, logits=False):
        import keras.backend as k
        k.set_learning_phase(0)

        if len(self._output.shape) == 2:
            nb_outputs = self._output.shape[1]
        else:
            raise ValueError('Unexpected output shape for classification in Keras model.')

        if label is None:
            logger.debug('Computing class gradients for all %i classes.', self.nb_classes)
            if logits:
                if not hasattr(self, '_class_grads_logits'):
                    class_grads_logits = [k.gradients(self._preds_op[:, i], input_tensor)[0]
                                          for i in range(nb_outputs)]
                    self._class_grads_logits = k.function([input_tensor], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads'):
                    class_grads = [k.gradients(k.softmax(self._preds_op)[:, i], input_tensor)[0]
                                   for i in range(nb_outputs)]
                    self._class_grads = k.function([input_tensor], class_grads)

        else:
            if type(label) is int:
                unique_labels = [label]
                logger.debug('Computing class gradients for class %i.', label)
            else:
                unique_labels = np.unique(label)
                logger.debug('Computing class gradients for classes %s.', str(unique_labels))

            if logits:
                if not hasattr(self, '_class_grads_logits_idx'):
                    self._class_grads_logits_idx = [None for _ in range(nb_outputs)]

                for l in unique_labels:
                    if self._class_grads_logits_idx[l] is None:
                        class_grads_logits = [k.gradients(self._preds_op[:, l], input_tensor)[0]]
                        self._class_grads_logits_idx[l] = k.function([input_tensor], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads_idx'):
                    self._class_grads_idx = [None for _ in range(nb_outputs)]

                for l in unique_labels:
                    if self._class_grads_idx[l] is None:
                        class_grads = [k.gradients(k.softmax(self._preds_op)[:, l], input_tensor)[0]]
                        self._class_grads_idx[l] = k.function([input_tensor], class_grads)

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        from keras.layers import InputLayer, Embedding

        layer_names = [layer.name for layer in self._model.layers[:-1]
                       if not isinstance(layer, InputLayer) and not isinstance(layer, Embedding)]
        logger.info('Inferred %i hidden layers on Keras classifier.', len(layer_names))

        return layer_names


class KerasImageClassifier(ImageClassifier, KerasClassifier):
    def __init__(self, clip_values, model, use_logits=False, channel_index=3, defences=None, preprocessing=(0, 1),
                 input_layer=0, output_layer=0, custom_activation=False):
        """
        Create a :class:`KerasImageClassifier` instance from a Keras model. Assumes the `model` passed as argument is
        compiled.

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
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param input_layer: Which layer to consider as the input when the model has multiple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the output when the model has multiple output layers.
        :type output_layer: `int`
        :param custom_activation: True if the model uses the last activation other than softmax and requires to use the
               output probability rather than the logits by attacks.
        :type custom_activation: `bool`
        """
        import keras.backend as k

        ImageClassifier.__init__(self, clip_values=clip_values, channel_index=channel_index, defences=defences,
                                 preprocessing=preprocessing)

        KerasClassifier.__init__(self, model=model, use_logits=use_logits, input_layer=input_layer,
                                 output_layer=output_layer, custom_activation=custom_activation)

        self._input_shape = k.int_shape(self._input)[1:]
        self._channel_index = channel_index

    def _init_class_grads(self, label=None, logits=False):
        self._gen_init_class_grads(input_tensor=self._input, label=label, logits=logits)


class KerasTextClassifier(TextClassifier, KerasClassifier):
    """
    Class providing an implementation for integrating text models from Keras.
    """
    def __init__(self, model, ids, use_logits=False, embedding_layer=0, input_layer=0, output_layer=0,
                 custom_activation=False):
        """
        Create a :class:`KerasTextClassifier` instance from a Keras model. Assumes the `model` passed as argument is
        compiled.

        :param model: Keras model
        :type model: `keras.models.Model`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param embedding_layer: Which layer to consider as providing the embedding of the vocabulary. When using the
               Keras `Embedding` layer, this can only be the first layer in the model.
        :type embedding_layer: `int`
        :param input_layer: Which layer to consider as the input when the model has multiple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the output when the model has multiple output layers.
        :type output_layer: `int`
        :param custom_activation: True if the model uses the last activation other than softmax and requires to use the
               output probability rather than the logits by attacks.
        :type custom_activation: `bool`
        """
        import keras.backend as k

        TextClassifier.__init__(self)
        KerasClassifier.__init__(self, model=model, use_logits=use_logits, input_layer=input_layer,
                                 output_layer=output_layer, custom_activation=custom_activation)

        if type(embedding_layer) is int:
            embedding = self._model.layers[embedding_layer]
        else:
            raise ValueError('Expected `int` for `embedding_layer`, got %s.' % str(type(embedding_layer)))

        self._embedding = embedding
        self._embedding_from_input = k.function([self._input], [self._embedding.output])
        self._preds_from_embedding = k.function([self._embedding.output], [self._preds_op])
        self._ids = ids

        if not hasattr(self, '_loss_grads') or self._loss_grads is None:
            loss_grads = k.gradients(self._loss_op, self._embedding.output)

            if k.backend() == 'tensorflow':
                loss_grads = loss_grads[0]
            elif k.backend() == 'cntk':
                raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

            self._loss_grads = k.function([self._input, self._label_op], [loss_grads])

    def predict_from_embedding(self, x_emb, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs in embedding form.

        :param x_emb: Array of inputs in embedding form, often shaped as `(batch_size, input_length, embedding_size)`.
        :type x_emb: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        # Run predictions with batching
        preds = np.zeros((x_emb.shape[0], *self._output.shape[1:]), dtype=np.float32)
        for b in range(int(np.ceil(x_emb.shape[0] / float(batch_size)))):
            begin, end = b * batch_size,  min((b + 1) * batch_size, x_emb.shape[0])
            preds[begin:end] = self._preds_from_embedding([x_emb[begin:end]])[0]

            if not logits:
                exp = np.exp(preds[begin:end] - np.max(preds[begin:end], axis=1, keepdims=True))
                preds[begin:end] = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def to_embedding(self, x):
        """
        Convert the received classifier input `x` from token (words or characters) indices to embeddings.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Embedding form of sample `x`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        return self._embedding_from_input([x])[0]

    def to_id(self, x_emb, strategy='nearest', metric='cosine'):
        """
        Convert the received input from embedding space to classifier input (most often, token indices).

        :param x_emb: Array of inputs in embedding form, often shaped as `(batch_size, input_length, embedding_size)`.
        :type x_emb: `np.ndarray`
        :param strategy: Strategy from mapping from embedding space back to input space.
        :type strategy: `str` or `Callable`
        :param metric: Metric to be used in the embedding space when determining vocabulary token proximity.
        :type metric: `str` or `Callable`
        :return: Array of token indices for sample `x_emb`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        if strategy != 'nearest':
            raise ValueError('Nearest neighbor is currently the only supported strategy for mapping embeddings to '
                             'valid tokens.')

        if metric == 'cosine':
            from art.utils import cosine

            v_size = len(self._ids)
            if v_size % x_emb.shape[1] > 0:
                for _ in range(x_emb.shape[1] - (v_size % x_emb.shape[1])):
                    self._ids.append(self._ids[0])
            embeddings = self.to_embedding(np.reshape(np.array(self._ids), (-1, x_emb.shape[1])))
            embeddings = np.reshape(embeddings, (-1, x_emb.shape[2]))[:v_size]
            self._ids = self._ids[:v_size]

            neighbors = []
            for x in x_emb:
                for emb_x in x:
                    metric = [cosine(emb, emb_x) for emb in embeddings]
                    neighbors.append(self._ids[int(np.argpartition(metric, -1)[-1])])
        else:
            raise ValueError('Cosine similarity is currently the only supported metric for mapping embeddings to '
                             'valid tokens.')

        return np.reshape(np.array(neighbors), x_emb.shape[:-1])

    def _init_class_grads(self, label=None, logits=False):
        self._gen_init_class_grads(input_tensor=self._embedding, label=label, logits=logits)


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
