# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements (De)Randomized Smoothing for Certifiable Defense against Patch Attacks

| Paper link: https://arxiv.org/abs/2002.10733
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import importlib
import logging
from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import random

import numpy as np
from tqdm import tqdm

from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.derandomized_smoothing.vision_transformers.pytorch import PyTorchSmoothedViT
from art.estimators.certification.derandomized_smoothing.derandomized_smoothing import DeRandomizedSmoothingMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    import torchvision
    from timm.models.vision_transformer import VisionTransformer
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchDeRandomizedSmoothingCNN(DeRandomizedSmoothingMixin, PyTorchClassifier):
    """
    Implementation of (De)Randomized Smoothing applied to classifier predictions as introduced
    in Levine et al. (2020).

    | Paper link: https://arxiv.org/abs/2002.10733
    """

    estimator_params = PyTorchClassifier.estimator_params + ["ablation_type", "ablation_size", "threshold", "logits"]

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        ablation_type: str,
        ablation_size: int,
        threshold: float,
        logits: bool,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
    ):
        """
        Create a derandomized smoothing classifier.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param ablation_type: The type of ablation to perform, must be either "column" or "block"
        :param ablation_size: The size of the data portion to retain after ablation. Will be a column of size N for
                              "column" ablation type or a NxN square for ablation of type "block"
        :param threshold: The minimum threshold to count a prediction.
        :param logits: if the model returns logits or normalized probabilities
        :param optimizer: The optimizer used to train the classifier.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
            ablation_type=ablation_type,
            ablation_size=ablation_size,
            threshold=threshold,
            logits=logits,
        )

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        import torch

        x = x.astype(ART_NUMPY_DTYPE)
        outputs = PyTorchClassifier.predict(self, x=x, batch_size=batch_size, training_mode=training_mode, **kwargs)

        if not self.logits:
            return np.asarray((outputs >= self.threshold))
        return np.asarray(
            (torch.nn.functional.softmax(torch.from_numpy(outputs), dim=1) >= self.threshold).type(torch.int)
        )

    def predict(
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:  # type: ignore
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Input samples.
        :param batch_size: Batch size.
        :param training_mode: if to run the classifier in training mode
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        return DeRandomizedSmoothingMixin.predict(self, x, batch_size=batch_size, training_mode=training_mode, **kwargs)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        x = x.astype(ART_NUMPY_DTYPE)
        return PyTorchClassifier.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)


class PyTorchDeRandomizedSmoothing(PyTorchDeRandomizedSmoothingCNN, PyTorchSmoothedViT):
    """
    Interface class for the two De-randomized smoothing approaches supported by ART for pytorch.

    If a regular pytorch neural network is fed in then (De)Randomized Smoothing as introduced in Levine et al. (2020)
    is used.

    Otherwise, if a timm vision transfomer is fed in then Certified Patch Robustness via Smoothed Vision Transformers
    as introduced in Salman et al. (2021) is used.
    """

    def __init__(self, model: Union[str, "VisionTransformer", "torch.nn.Module"], **kwargs):
        import torch

        self.mode = None
        if importlib.util.find_spec("timm") is not None:
            from timm.models.vision_transformer import VisionTransformer

            if isinstance(model, (VisionTransformer, str)):
                PyTorchSmoothedViT.__init__(self, model, **kwargs)
                self.mode = "ViT"
            else:
                if isinstance(model, torch.nn.Module):
                    PyTorchDeRandomizedSmoothingCNN.__init__(self, model, **kwargs)
                    self.mode = "CNN"

        elif isinstance(model, torch.nn.Module):
            PyTorchDeRandomizedSmoothingCNN.__init__(self, model, **kwargs)
            self.mode = "CNN"

        if self.mode is None:
            raise ValueError("Model type not recognized.")

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        drop_last: bool = False,
        scheduler: Optional[Any] = None,
        update_batchnorm: bool = True,
        batchnorm_update_epochs: int = 1,
        transform: Optional["torchvision.transforms.transforms.Compose"] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param update_batchnorm: ViT specific argument.
                                 If to run the training data through the model to update any batch norm statistics prior
                                 to training. Useful on small datasets when using pre-trained ViTs.
        :param batchnorm_update_epochs: ViT specific argument. How many times to forward pass over the training data
                                        to pre-adjust the batchnorm statistics.
        :param transform: ViT specific argument. Torchvision compose of relevant augmentation transformations to apply.
        :param verbose: if to display training progress bars
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch

        # Set model mode
        self._model.train(mode=training_mode)

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        if update_batchnorm and self.mode == "ViT":  # VIT specific
            self.update_batchnorm(x_preprocessed, batch_size, nb_epochs=batchnorm_update_epochs)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = len(x_preprocessed) / float(batch_size)
        if drop_last:
            num_batch = int(np.floor(num_batch))
        else:
            num_batch = int(np.ceil(num_batch))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in tqdm(range(nb_epochs)):
            # Shuffle the examples
            random.shuffle(ind)

            epoch_acc = []
            epoch_loss = []
            epoch_batch_sizes = []

            pbar = tqdm(range(num_batch), disable=not verbose)

            # Train for one epoch
            for m in pbar:
                i_batch = np.copy(x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]])
                i_batch = self.ablator.forward(i_batch)

                if transform is not None and self.mode == "ViT":  # VIT specific
                    i_batch = transform(i_batch)

                if isinstance(i_batch, np.ndarray):
                    i_batch = torch.from_numpy(i_batch).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                try:
                    model_outputs = self.model(i_batch)
                except ValueError as err:
                    if "Expected more than 1 value per channel when training" in str(err):
                        logger.exception(
                            "Try dropping the last incomplete batch by setting drop_last=True in "
                            "method PyTorchClassifier.fit."
                        )
                    raise err

                loss = self.loss(model_outputs, o_batch)
                acc = self.get_accuracy(preds=model_outputs, labels=o_batch)

                # Do training
                if self._use_amp:  # pragma: no cover
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self.optimizer.step()

                epoch_acc.append(acc)
                epoch_loss.append(loss.cpu().detach().numpy())
                epoch_batch_sizes.append(len(i_batch))

                if verbose:
                    pbar.set_description(
                        f"Loss {np.average(epoch_loss, weights=epoch_batch_sizes):.3f} "
                        f"Acc {np.average(epoch_acc, weights=epoch_batch_sizes):.3f} "
                    )

            if scheduler is not None:
                scheduler.step()

    @staticmethod
    def get_accuracy(preds: Union[np.ndarray, "torch.Tensor"], labels: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Helper function to get the accuracy during training.

        :param preds: model predictions.
        :param labels: ground truth labels (not one hot).
        :return: prediction accuracy.
        """
        if not isinstance(preds, np.ndarray):
            preds = preds.detach().cpu().numpy()

        if not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()

        return np.sum(np.argmax(preds, axis=1) == labels) / len(labels)
