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
This module implements the base class `DetectorClassifier` for classifier and detector combinations.

Paper link:
    https://arxiv.org/abs/1705.07263
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np

from art.estimators.classification.classifier import ClassifierNeuralNetwork

if TYPE_CHECKING:
    from art.utils import PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class DetectorClassifier(ClassifierNeuralNetwork):
    """
    This class implements a Classifier extension that wraps a classifier and a detector.
    More details in https://arxiv.org/abs/1705.07263
    """

    estimator_params = ClassifierNeuralNetwork.estimator_params + ["classifier", "detector"]

    def __init__(
        self,
        classifier: ClassifierNeuralNetwork,
        detector: ClassifierNeuralNetwork,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Initialization for the DetectorClassifier.

        :param classifier: A trained classifier.
        :param detector: A trained detector applied for the binary classification.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier. Not applicable
               in this classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one. Not applicable in this classifier.
        """
        if preprocessing_defences is not None:
            raise NotImplementedError("Preprocessing is not applicable in this classifier.")

        super().__init__(
            model=None,
            clip_values=classifier.clip_values,
            preprocessing=preprocessing,
            channels_first=classifier.channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
        )

        self.classifier = classifier
        self.detector = detector
        self._nb_classes = classifier.nb_classes + 1
        self._input_shape = classifier.input_shape

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Compute the prediction logits
        classifier_outputs = self.classifier.predict(x=x, batch_size=batch_size)
        detector_outputs = self.detector.predict(x=x, batch_size=batch_size)
        detector_outputs = (np.reshape(detector_outputs, [-1]) + 1) * np.max(classifier_outputs, axis=1)
        detector_outputs = np.reshape(detector_outputs, [-1, 1])
        combined_outputs = np.concatenate([classifier_outputs, detector_outputs], axis=1)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=combined_outputs, fit=False)

        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :raises `NotImplementedException`: This method is not supported for detector-classifiers.
        """
        raise NotImplementedError

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :raises `NotImplementedException`: This method is not supported for detector-classifiers.
        """
        raise NotImplementedError

    def class_gradient(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        label: Union[int, List[int], np.ndarray, None] = None,
        training_mode: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        if not (  # pragma: no cover
            (label is None)
            or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self.nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Compute the gradient and return
        if label is None:
            combined_grads = self._compute_combined_grads(x, label=None)

        elif isinstance(label, (int, np.int)):
            if label < self.nb_classes - 1:
                # Compute and return from the classifier gradients
                combined_grads = self.classifier.class_gradient(x=x, label=label, training_mode=training_mode, **kwargs)

            else:
                # First compute the classifier gradients
                classifier_grads = self.classifier.class_gradient(
                    x=x, label=None, training_mode=training_mode, **kwargs
                )

                # Then compute the detector gradients
                detector_grads = self.detector.class_gradient(x=x, label=0, training_mode=training_mode, **kwargs)

                # Chain the detector gradients for the first component
                classifier_preds = self.classifier.predict(x=x)
                maxind_classifier_preds = np.argmax(classifier_preds, axis=1)
                max_classifier_preds = classifier_preds[np.arange(x.shape[0]), maxind_classifier_preds]
                first_detector_grads = max_classifier_preds[:, None, None, None, None] * detector_grads

                # Chain the detector gradients for the second component
                max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_preds]
                detector_preds = self.detector.predict(x=x)
                second_detector_grads = max_classifier_grads * (detector_preds + 1)[:, None, None]
                second_detector_grads = second_detector_grads[None, ...]
                second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)

                # Update detector gradients
                combined_grads = first_detector_grads + second_detector_grads

        else:
            # Compute indexes for classifier labels and detector labels
            classifier_idx = np.where(label < self.nb_classes - 1)
            detector_idx = np.where(label == self.nb_classes - 1)

            # Initialize the combined gradients
            combined_grads = np.zeros(shape=(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))

            # First compute the classifier gradients for classifier_idx
            if classifier_idx:
                combined_grads[classifier_idx] = self.classifier.class_gradient(
                    x=x[classifier_idx], label=label[classifier_idx], training_mode=training_mode, **kwargs
                )

            # Then compute the detector gradients for detector_idx
            if detector_idx:
                # First compute the classifier gradients for detector_idx
                classifier_grads = self.classifier.class_gradient(
                    x=x[detector_idx], label=None, training_mode=training_mode, **kwargs
                )

                # Then compute the detector gradients for detector_idx
                detector_grads = self.detector.class_gradient(
                    x=x[detector_idx], label=0, training_mode=training_mode, **kwargs
                )

                # Chain the detector gradients for the first component
                classifier_preds = self.classifier.predict(x=x[detector_idx])
                maxind_classifier_preds = np.argmax(classifier_preds, axis=1)
                max_classifier_preds = classifier_preds[np.arange(len(detector_idx)), maxind_classifier_preds]
                first_detector_grads = max_classifier_preds[:, None, None, None, None] * detector_grads

                # Chain the detector gradients for the second component
                max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_preds]
                detector_preds = self.detector.predict(x=x[detector_idx])
                second_detector_grads = max_classifier_grads * (detector_preds + 1)[:, None, None]
                second_detector_grads = second_detector_grads[None, ...]
                second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)

                # Update detector gradients
                detector_grads = first_detector_grads + second_detector_grads

                # Reassign the combined gradients
                combined_grads[detector_idx] = detector_grads

        return combined_grads

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    def loss_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        :raises `NotImplementedException`: This method is not supported for detector-classifiers.
        """
        raise NotImplementedError

    @property
    def layer_names(self) -> List[str]:
        """
        Return the hidden layers in the model, if applicable. This function is not supported for the
        Classifier and Detector classes.

        :return: The hidden layers in the model, input and output layers excluded.
        :raises `NotImplementedException`: This method is not supported for detector-classifiers.
        """
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int = 128, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations.
        :param batch_size: Size of batches.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :raises `NotImplementedException`: This method is not supported for detector-classifiers.
        """
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        self.classifier.save(filename=filename + "_classifier", path=path)
        self.detector.save(filename=filename + "_detector", path=path)

    def __repr__(self):
        repr_ = "%s(classifier=%r, detector=%r, postprocessing_defences=%r, " "preprocessing=%r)" % (
            self.__module__ + "." + self.__class__.__name__,
            self.classifier,
            self.detector,
            self.postprocessing_defences,
            self.preprocessing,
        )

        return repr_

    def _compute_combined_grads(
        self, x: np.ndarray, label: Union[int, List[int], np.ndarray, None] = None
    ) -> np.ndarray:
        # Compute the classifier gradients
        classifier_grads = self.classifier.class_gradient(x=x, label=label)

        # Then compute the detector gradients
        detector_grads = self.detector.class_gradient(x=x, label=label)

        # Chain the detector gradients for the first component
        classifier_preds = self.classifier.predict(x=x)
        maxind_classifier_preds = np.argmax(classifier_preds, axis=1)
        max_classifier_preds = classifier_preds[np.arange(classifier_preds.shape[0]), maxind_classifier_preds]
        first_detector_grads = max_classifier_preds[:, None, None, None, None] * detector_grads

        # Chain the detector gradients for the second component
        max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_preds]
        detector_preds = self.detector.predict(x=x)
        second_detector_grads = max_classifier_grads * (detector_preds + 1)[:, None, None]
        second_detector_grads = second_detector_grads[None, ...]
        second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)

        # Update detector gradients
        detector_grads = first_detector_grads + second_detector_grads

        # Combine the gradients
        combined_logits_grads = np.concatenate([classifier_grads, detector_grads], axis=1)

        return combined_logits_grads
