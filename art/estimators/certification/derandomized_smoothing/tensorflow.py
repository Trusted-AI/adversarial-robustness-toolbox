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

import logging
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from art.estimators.certification.derandomized_smoothing.derandomized import DeRandomizedSmoothingMixin
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE, ABLATOR_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowV2DeRandomizedSmoothing(TensorFlowV2Classifier, DeRandomizedSmoothingMixin):
    """
    Implementation of (De)Randomized Smoothing applied to classifier predictions as introduced
    in Levine et al. (2020).

    | Paper link: https://arxiv.org/abs/2002.10733
    """

    estimator_params = TensorFlowV2Classifier.estimator_params + [
        "ablation_type",
        "ablation_size",
        "threshold",
        "logits",
    ]

    def __init__(
        self,
        model,
        nb_classes: int,
        ablation_type: str,
        ablation_size: int,
        threshold: float,
        logits: bool,
        input_shape: Tuple[int, ...],
        loss_object: Optional["tf.Tensor"] = None,
        optimizer: Optional["tf.keras.optimizers.legacy.Optimizer"] = None,
        train_step: Optional[Callable] = None,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ):
        """
        Create a derandomized smoothing classifier.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :type model: `function` or `callable class`
        :param nb_classes: the number of classes in the classification task.
        :param ablation_type: The type of ablation to perform, must be either "column" or "block"
        :param ablation_size: The size of the data portion to retain after ablation. Will be a column of size N for
                              "column" ablation type or a NxN square for ablation of type "block"
        :param threshold: The minimum threshold to count a prediction.
        :param logits: if the model returns logits or normalized probabilities
        :param input_shape: Shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
               the model and computing gradients of the loss w.r.t. the input.
        :param optimizer: The optimizer used to train the classifier.
        :param train_step: A function that applies a gradient update to the trainable variables with signature
               `train_step(model, images, labels)`. This will override the default training loop that uses the
               provided `loss_object` and `optimizer` parameters. It is recommended to use the `@tf.function`
               decorator, if possible, for efficient training.
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
        """
        # input channels are internally doubled for the certification algorithm.
        input_shape = (input_shape[0], input_shape[1], input_shape[2] * 2)
        super().__init__(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            loss_object=loss_object,
            optimizer=optimizer,
            train_step=train_step,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self.ablation_type = ablation_type
        self.logits = logits
        self.threshold = threshold
        self._channels_first = channels_first

        from art.estimators.certification.derandomized_smoothing.ablators.tensorflow import (
            ColumnAblator,
            BlockAblator,
        )

        if TYPE_CHECKING:
            self.ablator: ABLATOR_TYPE  # pylint: disable=used-before-assignment

        if self.ablation_type in {"column", "row"}:
            row_ablation_mode = self.ablation_type == "row"
            self.ablator = ColumnAblator(
                ablation_size=ablation_size, channels_first=self._channels_first, row_ablation_mode=row_ablation_mode
            )
        elif self.ablation_type == "block":
            self.ablator = BlockAblator(ablation_size=ablation_size, channels_first=self._channels_first)
        else:
            raise ValueError("Ablation type not supported. Must be either column or block")

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        import tensorflow as tf

        outputs = TensorFlowV2Classifier.predict(
            self, x=x, batch_size=batch_size, training_mode=training_mode, **kwargs
        )
        if self.logits:
            outputs = tf.nn.softmax(outputs)
        return np.asarray(outputs >= self.threshold).astype(int)

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Labels, one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param verbose: Display training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. This parameter currently only supports
                       "scheduler" which is an optional function that will be called at the end of every
                       epoch to adjust the learning rate.
        """
        import tensorflow as tf

        if self._train_step is None:  # pragma: no cover
            if self._loss_object is None:  # pragma: no cover
                raise TypeError(
                    "A loss function `loss_object` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )
            if self._optimizer is None:  # pragma: no cover
                raise ValueError(
                    "An optimizer `optimizer` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )

            @tf.function
            def train_step(model, images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = self.loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss, predictions

        else:
            train_step = self._train_step

        scheduler = kwargs.get("scheduler")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        for epoch in tqdm(range(nb_epochs), desc="Epochs"):
            num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

            epoch_acc = []
            epoch_loss = []
            epoch_batch_sizes = []

            pbar = tqdm(range(num_batch), disable=not verbose)

            ind = np.arange(len(x_preprocessed))
            for m in pbar:
                i_batch = np.copy(x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]])
                labels = y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]
                images = self.ablator.forward(i_batch)

                if self._train_step is None:
                    loss, predictions = train_step(self.model, images, labels)
                    acc = np.sum(np.argmax(predictions.numpy(), axis=1) == np.argmax(labels, axis=1)) / len(labels)
                    epoch_acc.append(acc)
                    epoch_loss.append(loss.numpy())
                    epoch_batch_sizes.append(len(i_batch))
                else:
                    train_step(self.model, images, labels)

                if verbose:
                    if self._train_step is None:
                        pbar.set_description(
                            f"Loss {np.average(epoch_loss, weights=epoch_batch_sizes):.3f} "
                            f"Acc {np.average(epoch_acc, weights=epoch_batch_sizes):.3f} "
                        )
                    else:
                        pbar.set_description("Batches")

            if scheduler is not None:
                scheduler(epoch)

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Performs cumulative predictions over every ablation location

        :param x: Unablated image
        :param batch_size: the batch size for the prediction
        :param training_mode: if to run the classifier in training mode
        :return: cumulative predictions after sweeping over all the ablation configurations.
        """
        if self._channels_first:
            columns_in_data = x.shape[-1]
            rows_in_data = x.shape[-2]
        else:
            columns_in_data = x.shape[-2]
            rows_in_data = x.shape[-3]

        if self.ablation_type in {"column", "row"}:
            if self.ablation_type == "column":
                ablate_over_range = columns_in_data
            else:
                # image will be transposed, so loop over the number of rows
                ablate_over_range = rows_in_data

            for ablation_start in range(ablate_over_range):
                ablated_x = self.ablator.forward(np.copy(x), column_pos=ablation_start)
                if ablation_start == 0:
                    preds = self._predict_classifier(
                        ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                    )
                else:
                    preds += self._predict_classifier(
                        ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                    )
        elif self.ablation_type == "block":
            for xcorner in range(rows_in_data):
                for ycorner in range(columns_in_data):
                    ablated_x = self.ablator.forward(np.copy(x), row_pos=xcorner, column_pos=ycorner)
                    if ycorner == 0 and xcorner == 0:
                        preds = self._predict_classifier(
                            ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                        )
                    else:
                        preds += self._predict_classifier(
                            ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                        )
        return preds

    def eval_and_certify(
        self,
        x: np.ndarray,
        y: np.ndarray,
        size_to_certify: int,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """
        Evaluates the normal and certified performance over the supplied data.

        :param x: Evaluation data.
        :param y: Evaluation labels.
        :param size_to_certify: The size of the patch to certify against.
                                If not provided will default to the ablation size.
        :param batch_size: batch size when evaluating.
        :param verbose: If to display the progress bar
        :return: The accuracy and certified accuracy over the dataset
        """
        import tensorflow as tf

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        pbar = tqdm(range(num_batch), disable=not verbose)
        accuracy = tf.constant(np.array(0.0), dtype=tf.dtypes.int32)
        cert_sum = tf.constant(np.array(0.0), dtype=tf.dtypes.int32)
        n_samples = 0

        for m in pbar:
            if m == (num_batch - 1):
                i_batch = np.copy(x_preprocessed[m * batch_size :])
                o_batch = y_preprocessed[m * batch_size :]
            else:
                i_batch = np.copy(x_preprocessed[m * batch_size : (m + 1) * batch_size])
                o_batch = y_preprocessed[m * batch_size : (m + 1) * batch_size]

            pred_counts = self.predict(i_batch)

            _, cert_and_correct, top_predicted_class = self.ablator.certify(
                pred_counts, size_to_certify=size_to_certify, label=o_batch
            )
            cert_sum += tf.math.reduce_sum(tf.where(cert_and_correct, 1, 0))
            accuracy += tf.math.reduce_sum(tf.where(top_predicted_class == np.argmax(o_batch, axis=-1), 1, 0))
            n_samples += len(cert_and_correct)

            pbar.set_description(f"Normal Acc {accuracy / n_samples:.3f} " f"Cert Acc {cert_sum / n_samples:.3f}")
        return (accuracy / n_samples), (cert_sum / n_samples)
