from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random

import numpy as np
import six

from art.classifiers.classifier import Classifier

logger = logging.getLogger(__name__)


class DetectorClassifier(Classifier):
    """
    This class implements a Classifier extension that wraps a classifier and a detector.
    More details in https://arxiv.org/abs/1705.07263
    """
    def __init__(self, classifier, detector, defences=None, preprocessing=(0, 1)):
        """
        Initialization for the DetectorClassifier.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param detector: A trained detector applied for the binary classification.
        :type detector: `art.detection.detector.Detector`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(DetectorClassifier, self).__init__(clip_values=classifier.clip_values, preprocessing=preprocessing,
                                                 channel_index=classifier.channel_index, defences=defences)

        self.classifier = classifier
        self.detector = detector
        self._nb_classes = classifier.nb_classes
        self._input_shape = classifier.input_shape

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

        # Run prediction with batch processing
        results = np.zeros((x_.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_.shape[0])

            model_outputs = self._model(torch.from_numpy(x_[begin:end]).to(self._device).float())
            (logit_output, output) = (model_outputs[-2], model_outputs[-1])

            if logits:
                results[begin:end] = logit_output.detach().cpu().numpy()
            else:
                results[begin:end] = output.detach().cpu().numpy()

        return results

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
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
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: `DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

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
        pass

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
        import torch

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(self._apply_processing(x)).to(self._device)
        inputs_t = inputs_t.float()
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(np.argmax(y, axis=1)).to(self._device)

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        loss = self._loss(model_outputs[-1], labels_t)

        # Clean gradients
        self._model.zero_grad()
        # inputs_t.grad.data.zero_()

        # Compute gradients
        loss.backward()
        grds = inputs_t.grad.cpu().numpy().copy()
        grds = self._apply_processing_gradient(grds)
        assert grds.shape == x.shape

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
        import torch

        # Apply defences
        x_ = self._apply_processing(x)
        x_ = self._apply_defences_predict(x_)

        # Get index of the extracted layer
        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError("Layer name %s not supported" % layer)
            layer_index = self._layer_names.index(layer)

        elif isinstance(layer, (int, np.integer)):
            layer_index = layer

        else:
            raise TypeError("Layer must be of type str or int")

        # Run prediction with batch processing
        results = []
        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_.shape[0])

            # Run prediction for the current batch
            layer_output = self._model(torch.from_numpy(x_[begin:end]).to(self._device).float())[layer_index]
            results.append(layer_output.detach().cpu().numpy())

        results = np.concatenate(results)

        return results

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
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os
        import torch

        if path is None:
            from art import DATA_PATH
            full_path = os.path.join(DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self._model.state_dict(), full_path + '.model')
        torch.save(self._optimizer.state_dict(), full_path + '.optimizer')
        logger.info("Model state dict saved in path: %s.", full_path + '.model')
        logger.info("Optimizer state dict saved in path: %s.", full_path + '.optimizer')

    # def _forward_at(self, inputs, layer):
    #     """
    #     Compute the forward at a specific layer.
    #
    #     :param inputs: Input data.
    #     :type inputs: `np.ndarray`
    #     :param layer: The layer where to get the forward results.
    #     :type layer: `int`
    #     :return: The forward results at the layer.
    #     :rtype: `torch.Tensor`
    #     """
    #     print(layer)
    #     results = inputs
    #     for l in list(self._model.modules())[1:layer + 2]:
    #         print(l)
    #
    #         results = l(results)
    #
    #         print(results.shape)
    #
    #     return results

    def _make_model_wrapper(self, model):
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        try:
            import torch.nn as nn

            # Define model wrapping class only if not defined before
            if not hasattr(self, '_ModelWrapper'):

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
                        if isinstance(self._model, nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
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
                                     The intended order of the layers tries to match their order in the model, but this
                                     is not guaranteed either. In addition, the function can only infer the internal
                                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                                     return the logit layer.
                        """
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            for name, module_ in self._model._modules.items():
                                result.append(name + "_" + str(module_))

                        elif isinstance(self._model, nn.Module):
                            result.append("logit_layer")

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")
                        logger.info('Inferred %i hidden layers on PyTorch classifier.', len(result))

                        return result

                # Set newly created class as private attribute
                self._ModelWrapper = ModelWrapper

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._ModelWrapper(model)

        except ImportError:
            raise ImportError('Could not find PyTorch (`torch`) installation.')
