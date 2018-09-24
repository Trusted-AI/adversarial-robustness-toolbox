from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random
import six

from art.classifiers.classifier import Classifier, ImageClassifier, TextClassifier


class PyTorchClassifier(Classifier):
    """
    This class implements a classifier with the PyTorch framework.
    """
    def __init__(self, model, loss, optimizer, nb_classes):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param model: PyTorch model. The forward function of the model must return the logit output.
        :type model: is instance of `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :type loss: `torch.nn.modules.loss._Loss`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        """
        super(PyTorchClassifier, self).__init__()

        self._nb_classes = nb_classes
        self._model = PyTorchClassifier.ModelWrapper(model)
        self._loss = loss
        self._optimizer = optimizer

        # Get the internal layers
        self._layer_names = self._model.get_layers

        # Use GPU if possible
        import torch
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

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
        import torch

        # Apply defences
        x_ = self._apply_processing(x)
        x_ = self._apply_defences_predict(x_)

        # Set test phase
        self._model.train(False)

        # Run prediction with batch processing
        results = np.zeros((x_.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_.shape[0])

            model_outputs = self._model(torch.from_numpy(x_[begin:end]).to(self._device))
            (logit_output, output) = (model_outputs[-2], model_outputs[-1])

            if logits:
                results[begin:end] = logit_output.detach().cpu().numpy()
            else:
                results[begin:end] = output.detach().cpu().numpy()

        return results

    def fit(self, x, y, batch_size=128, nb_epochs=10):
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
        import torch

        # Apply defences
        x_ = self._apply_processing(x)
        x_, y_ = self._apply_defences_fit(x_, y)
        y_ = np.argmax(y_, axis=1)

        # Set train phase
        self._model.train(True)

        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        ind = np.arange(len(x_))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = torch.from_numpy(x_[ind[m * batch_size:(m + 1) * batch_size]]).to(self._device)
                o_batch = torch.from_numpy(y_[ind[m * batch_size:(m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Actual training
                model_outputs = self._model(i_batch)
                loss = self._loss(model_outputs[-1], o_batch)
                loss.backward()
                self._optimizer.step()

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int` or `numpy.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        import torch

        if not ((label is None) or (type(label) is int and label in range(self._nb_classes)) or (
            type(label) is np.ndarray and len(label.shape) == 1 and (label < self._nb_classes).all()
            and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % label)

        # Convert the inputs to Tensors
        x_ = torch.from_numpy(self._apply_processing(x)).to(self._device)

        # Compute the gradient and return
        # Run prediction
        model_outputs = self._model(x_)

        # Set where to get gradient from
        (logit_output, output) = (model_outputs[-2], model_outputs[-1])
        if logits:
            preds = logit_output
        else:
            preds = output

        # Compute the gradient
        grads = []

        self._model.zero_grad()
        if label is None:
            for i in range(self.nb_classes):
                torch.autograd.backward(preds[:, i], torch.Tensor([1.] * len(preds[:, 0])), retain_graph=True)

            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)

        elif type(label) is int:
            torch.autograd.backward(preds[:, label], torch.Tensor([1.] * len(preds[:, 0])), retain_graph=True)

            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)

        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(preds[:, i], torch.Tensor([1.] * len(preds[:, 0])), retain_graph=True)

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]
            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)

        return grads

    def loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import torch

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(self._apply_processing(x)).to(self._device)

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(np.argmax(y, axis=1)).to(self._device)

        # Compute gradient wrt what
        layer_idx = self._init_grads()
        if layer_idx < 0:
            inputs_t.requires_grad = True

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)

        # Set where to get gradient
        if layer_idx >= 0:
            input_grad = model_outputs[layer_idx]
        else:
            input_grad = inputs_t

        # Set where to get gradient from
        loss = self._loss(model_outputs[-1], labels_t)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()
            return hook
        input_grad.register_hook(save_grad())

        loss.backward()
        grds = self._apply_processing_gradient(grads[0])

        return grds

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either. In addition, the function can only infer the internal layers if the input
                     model is of type `nn.Sequential`, otherwise, it will only return the logit layer.
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
        import torch

        # Apply defences
        x = self._apply_defences_predict(x)

        # Set test phase
        self._model.train(False)

        # Run prediction
        model_outputs = self._model(torch.from_numpy(self._apply_processing(x)).to(self._device))[:-1]

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError("Layer name %s not supported" % layer)
            layer_index = self._layer_names.index(layer)

        elif isinstance(layer, (int, np.integer)):
            layer_index = layer

        else:
            raise TypeError("Layer must be of type str or int")

        return model_outputs[layer_index].detach().cpu().numpy()

    try:
        import torch.nn as nn

        class ModelWrapper(nn.Module):
            """
            This is a wrapper for the input model.
            """
            def __init__(self, model):
                """
                Initialization by storing the input model.

                :param model: PyTorch model. The forward function of the model must return the logit output.
                :type model: is instance of `torch.nn.Module`
                """
                super(PyTorchClassifier.ModelWrapper, self).__init__()
                self._model = model

            def forward(self, x):
                """
                This is where we get outputs from the input model.

                :param x: Input data.
                :type x: `torch.Tensor`
                :return: a list of output layers, where the last 2 layers are logit and final outputs.
                :rtype: `list`
                """
                import torch.nn as nn

                result = []
                if type(self._model) is nn.Sequential:
                    for _, module_ in self._model._modules.items():
                        x = module_(x)
                        #x.retain_grad()
                        result.append(x)

                elif isinstance(self._model, nn.Module):
                    x = self._model(x)
                    result.append(x)

                else:
                    raise TypeError("The input model must inherit from `nn.Module`.")

                output_layer = nn.functional.softmax(x, dim=1)
                result.append(output_layer)

                return result

            @property
            def get_layers(self):
                """
                Return the hidden layers in the model, if applicable.

                :return: The hidden layers in the model, input and output layers excluded.
                :rtype: `list`

                .. warning:: `get_layers` tries to infer the internal structure of the model.
                             This feature comes with no guarantees on the correctness of the result.
                             The intended order of the layers tries to match their order in the model, but this is not
                             guaranteed either. In addition, the function can only infer the internal layers if the
                             input model is of type `nn.Sequential`, otherwise, it will only return the logit layer.
                """
                import torch.nn as nn

                result = []
                if type(self._model) is nn.Sequential:
                    for name, module_ in self._model._modules.items():
                        result.append(name + "_" + str(module_))

                elif isinstance(self._model, nn.Module):
                    result.append("logit_layer")

                else:
                    raise TypeError("The input model must inherit from `nn.Module`.")

                return result

    except ImportError:
        raise ImportError('Could not find PyTorch (`torch`) installation.')


class PyTorchImageClassifier(ImageClassifier, PyTorchClassifier):
    def __init__(self, clip_values, model, loss, optimizer, input_shape, nb_classes, channel_index=1, defences=None,
                 preprocessing=(0, 1)):
        """
        Create a :class:`PyTorchImageClassifier` instance from a Pytorch model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: PyTorch model. The forward function of the model must return the logit output.
        :type model: is instance of `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :type loss: `torch.nn.modules.loss._Loss`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param input_shape: The shape of one input instance.
        :type input_shape: `tuple`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """

        ImageClassifier.__init__(self, clip_values=clip_values, channel_index=channel_index, defences=defences,
                                 preprocessing=preprocessing)
        PyTorchClassifier.__init__(self, model=model, loss=loss, optimizer=optimizer, nb_classes=nb_classes)

        self._input_shape = input_shape

    def _init_grads(self):
        return -1


class PyTorchTextClassifier(TextClassifier, PyTorchClassifier):
    def __init__(self, model, embedding_layer, ids, loss, optimizer, nb_classes):
        """
        Create a :class:`PyTorchTextClassifier` instance from a Pytorch model.

        :param model: PyTorch model. The forward function of the model must return the logit output.
        :type model: is instance of `torch.nn.Module`
        :param embedding_layer: Which layer to consider as providing the embedding of the vocabulary.
        :type embedding_layer: `int`
        :param ids: List of ids in the vocabulary.
        :type ids: `list`
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :type loss: `torch.nn.modules.loss._Loss`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        """
        TextClassifier.__init__(self)
        PyTorchClassifier.__init__(self, model=model, loss=loss, optimizer=optimizer, nb_classes=nb_classes)

        if type(embedding_layer) is not int:
            raise ValueError('Expected `int` for `embedding_layer`, got %s.' % str(type(embedding_layer)))

        if embedding_layer not in range(len(self._layer_names)):
            raise ValueError("Embedding layer %d is out of range (0 to %d included)." % (embedding_layer,
                                                                                         len(self._layer_names) - 1))

        self._embedding_layer = embedding_layer
        self._ids = ids

    def _init_grads(self):
        return self._embedding_layer

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
        # Convert to ids
        x_ids = self.to_id(x_emb)

        # Run prediction
        preds = self.predict(x=x_ids, logits=logits, batch_size=batch_size)

        return preds

    def to_embedding(self, x):
        """
        Convert the received classifier input `x` from token (words or characters) indices to embeddings.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Embedding form of sample `x`.
        :rtype: `np.ndarray`
        """
        import torch

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(self._apply_processing(x)).to(self._device)

        # Compute embeddings
        model_outputs = self._model(inputs_t)

        return model_outputs[self._embedding_layer].cpu().detach().numpy()

















