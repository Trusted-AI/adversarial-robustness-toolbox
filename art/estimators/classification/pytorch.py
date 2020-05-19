# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
"""
This module implements the classifier `PyTorchClassifier` for PyTorch models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random

import numpy as np
import six

from art.estimators.pytorch import PyTorchEstimator
from art.estimators.classification.classifier import ClassifierMixin, ClassGradientsMixin

logger = logging.getLogger(__name__)


class PyTorchClassifier(ClassGradientsMixin, ClassifierMixin, PyTorchEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements a classifier with the PyTorch framework.
    """

    def __init__(
        self,
        model,
        loss,
        input_shape,
        nb_classes,
        optimizer=None,
        channel_index=1,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1),
        device_type="gpu",
    ):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :type model: `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :type loss: `torch.nn.modules.loss._Loss`
        :param input_shape: The shape of one input instance.
        :type input_shape: `tuple`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :type device_type: `string`
        """
        super().__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._nb_classes = nb_classes
        self._input_shape = input_shape
        self._model = self._make_model_wrapper(model)
        self._loss = loss
        self._optimizer = optimizer
        self._learning_phase = None

        # Get the internal layers
        self._layer_names = self._model.get_layers

        # Set device
        import torch

        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        self._model.to(self._device)

        # Index of layer at which the class gradients should be calculated
        self._layer_idx_gradients = -1

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        if isinstance(loss, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
            self._reduce_labels = True
        else:
            self._reduce_labels = False

    @property
    def model(self):
        return self._model._model

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        import torch

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

            model_outputs = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            output = model_outputs[-1]
            results[begin:end] = output.detach().cpu().numpy()

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        import torch

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = torch.from_numpy(x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self._model(i_batch)

                # Form the loss function
                loss = self._loss(model_outputs[-1], o_batch)

                # Actual training
                loss.backward()
                self._optimizer.step()

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        import torch
        from art.data_generators import PyTorchDataGenerator

        # Train directly in PyTorch
        if (
            isinstance(generator, PyTorchDataGenerator)
            and (self.preprocessing_defences is None or self.preprocessing_defences == [])
            and self.preprocessing == (0, 1)
        ):
            for _ in range(nb_epochs):
                for i_batch, o_batch in generator.iterator:
                    if isinstance(i_batch, np.ndarray):
                        i_batch = torch.from_numpy(i_batch).to(self._device)
                    else:
                        i_batch = i_batch.to(self._device)

                    if isinstance(o_batch, np.ndarray):
                        o_batch = torch.argmax(torch.from_numpy(o_batch).to(self._device), dim=1)
                    else:
                        o_batch = torch.argmax(o_batch.to(self._device), dim=1)

                    # Zero the parameter gradients
                    self._optimizer.zero_grad()

                    # Perform prediction
                    model_outputs = self._model(i_batch)

                    # Form the loss function
                    loss = self._loss(model_outputs[-1], o_batch)

                    # Actual training
                    loss.backward()
                    self._optimizer.step()
        else:
            # Fit a generic data generator through the API
            super(PyTorchClassifier, self).fit_generator(generator, nb_epochs=nb_epochs)

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        import torch

        if not (
            (label is None)
            or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self._nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = torch.from_numpy(x_preprocessed).to(self._device)

        # Compute gradients
        if self._layer_idx_gradients < 0:
            x_preprocessed.requires_grad = True

        # Run prediction
        model_outputs = self._model(x_preprocessed)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_preprocessed

        # Set where to get gradient from
        preds = model_outputs[-1]

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self._model.zero_grad()
        if label is None:
            for i in range(self.nb_classes):
                torch.autograd.backward(
                    preds[:, i], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
                )

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
            )
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(
                    preds[:, i], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
                )

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import torch

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        loss = self._loss(model_outputs[-1], labels_t)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = inputs_t.grad.cpu().numpy().copy()
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads

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

    def get_activations(self, x, layer, batch_size=128, intermediate=False):
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
        import torch

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False)

        # Get index of the extracted layer
        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError("Layer name %s not supported" % layer)
            layer_index = self._layer_names.index(layer)

        elif isinstance(layer, (int, np.integer)):
            layer_index = layer

        else:
            raise TypeError("Layer must be of type str or int")

        if intermediate:
            return self._model(torch.from_numpy(x).to(self._device))[layer_index]

        # Run prediction with batch processing
        results = []
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

            # Run prediction for the current batch
            layer_output = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))[layer_index]
            results.append(layer_output.detach().cpu().numpy())

        results = np.concatenate(results)

        return results

    def custom_gradient(self, nn_function, tensors, input_values):
        """
        Returns the gradient of the nn_function with respect to model input

        :param nn_function: an intermediate tensor representation of the function to differentiate
        :type nn_function: a Pytorch tensor
        :param tensors: the tensors or variables to differentiate with respect to
        :type tensors: `list`
        :param input_values: the inputs to evaluate the gradient
        :type input_values: `list`
        :return: the gradient of the function w.r.t vars
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def get_input_layer(self):
        return self._input

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        if isinstance(train, bool):
            self._learning_phase = train
            self._model.train(train)

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os
        import torch

        if path is None:
            from art.config import ART_DATA_PATH

            full_path = os.path.join(ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        # pylint: disable=W0212
        # disable pylint because access to _modules required
        torch.save(self.model.state_dict(), full_path + ".model")
        torch.save(self._optimizer.state_dict(), full_path + ".optimizer")
        logger.info("Model state dict saved in path: %s.", full_path + ".model")
        logger.info("Optimizer state dict saved in path: %s.", full_path + ".optimizer")

    def __getstate__(self):
        """
        Use to ensure `PytorchClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """
        import time
        import copy

        # pylint: disable=W0212
        # disable pylint because access to _model required
        state = self.__dict__.copy()
        state["inner_model"] = copy.copy(state["_model"]._model)

        # Remove the unpicklable entries
        del state["_model_wrapper"]
        del state["_device"]
        del state["_model"]

        model_name = str(time.time())
        state["model_name"] = model_name
        self.save(model_name)

        return state

    def __setstate__(self, state):
        """
        Use to ensure `PytorchClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to Pytorch
        import os
        import torch
        from art.config import ART_DATA_PATH

        # Recover model
        full_path = os.path.join(ART_DATA_PATH, state["model_name"])
        model = state["inner_model"]
        model.load_state_dict(torch.load(str(full_path) + ".model"))
        model.eval()
        self._model = self._make_model_wrapper(model)

        # Recover device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Recover optimizer
        self._optimizer.load_state_dict(torch.load(str(full_path) + ".optimizer"))

        self.__dict__.pop("model_name", None)
        self.__dict__.pop("inner_model", None)

    def __repr__(self):
        repr_ = (
            "%s(model=%r, loss=%r, optimizer=%r, input_shape=%r, nb_classes=%r, channel_index=%r, "
            "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._model,
                self._loss,
                self._optimizer,
                self._input_shape,
                self.nb_classes,
                self.channel_index,
                self.clip_values,
                self.preprocessing_defences,
                self.postprocessing_defences,
                self.preprocessing,
            )
        )

        return repr_

    def _make_model_wrapper(self, model):
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        try:
            import torch.nn as nn

            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

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
                        super(ModelWrapper, self).__init__()
                        self._model = model

                    # pylint: disable=W0221
                    # disable pylint because of API requirements for function
                    def forward(self, x):
                        """
                        This is where we get outputs from the input model.

                        :param x: Input data.
                        :type x: `torch.Tensor`
                        :return: a list of output layers, where the last 2 layers are logit and final outputs.
                        :rtype: `list`
                        """
                        # pylint: disable=W0212
                        # disable pylint because access to _model required
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
                                result.append(x)

                        elif isinstance(self._model, nn.Module):
                            x = self._model(x)
                            result.append(x)

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")

                        return result

                    @property
                    def get_layers(self):
                        """
                        Return the hidden layers in the model, if applicable.

                        :return: The hidden layers in the model, input and output layers excluded.
                        :rtype: `list`

                        .. warning:: `get_layers` tries to infer the internal structure of the model.
                                     This feature comes with no guarantees on the correctness of the result.
                                     The intended order of the layers tries to match their order in the model, but this
                                     is not guaranteed either. In addition, the function can only infer the internal
                                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                                     return the logit layer.
                        """
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            # pylint: disable=W0212
                            # disable pylint because access to _modules required
                            for name, module_ in self._model._modules.items():
                                result.append(name + "_" + str(module_))

                        elif isinstance(self._model, nn.Module):
                            result.append("final_layer")

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")
                        logger.info("Inferred %i hidden layers on PyTorch classifier.", len(result))

                        return result

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._model_wrapper(model)

        except ImportError:
            raise ImportError("Could not find PyTorch (`torch`) installation.")
