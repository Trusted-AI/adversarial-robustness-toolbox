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
This module implements Sleeper Agent attack on Neural Networks.

| Paper link: https://arxiv.org/abs/2106.08970
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Tuple, TYPE_CHECKING, List, Union
import random

import numpy as np
from tqdm.auto import trange

from art.attacks.poisoning import GradientMatchingAttack

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class SleeperAgentAttack(GradientMatchingAttack):
    """
    Implementation of Sleeper Agent Attack

    | Paper link: https://arxiv.org/pdf/2106.08970.pdf
    """

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        percent_poison: float,
        patch: np.ndarray,
        indices_target: List[int],
        epsilon: float = 0.1,
        max_trials: int = 8,
        max_epochs: int = 250,
        learning_rate_schedule: Tuple[List[float], List[int]] = ([1e-1, 1e-2, 1e-3, 1e-4], [100, 150, 200, 220]),
        batch_size: int = 128,
        clip_values: Tuple[float, float] = (0, 1.0),
        verbose: int = 1,
        patching_strategy: str = "random",
        selection_strategy: str = "random",
        retraining_factor: int = 1,
        model_retrain: bool = False,
        model_retraining_epoch: int = 1,
        class_source: int = 0,
        class_target: int = 1,
        device_name: str = "cpu",
    ):
        """
        Initialize a Sleeper Agent poisoning attack.

        :param classifier: The proxy classifier used for the attack.
        :param percent_poison: The ratio of samples to poison among x_train, with range [0,1].
        :param patch: The patch to be applied as trigger.
        :param indices_target: The indices of training data having target label.
        :param epsilon: The L-inf perturbation budget.
        :param max_trials: The maximum number of restarts to optimize the poison.
        :param max_epochs: The maximum number of epochs to optimize the train per trial.
        :param learning_rate_schedule: The learning rate schedule to optimize the poison.
            A List of (learning rate, epoch) pairs. The learning rate is used
            if the current epoch is less than the specified epoch.
        :param batch_size: Batch size.
        :param clip_values: The range of the input features to the classifier.
        :param verbose: Show progress bars.
        :param patching_strategy: Patching strategy to be used for adding trigger, either random/fixed.
        :param selection_strategy: Selection strategy for getting the indices of
                             poison examples - either random/maximum gradient norm.
        :param retraining_factor: The factor for which retraining needs to be applied.
        :param model_retrain: True, if retraining has to be applied, else False.
        :param model_retraining_epoch: The epochs for which retraining has to be applied.
        :param class_source: The source class from which triggers were selected.
        :param class_target: The target label to which the poisoned model needs to misclassify.
        """
        clip_values_normalised = (classifier.clip_values - classifier.preprocessing.mean) / classifier.preprocessing.std
        clip_values_normalised = (clip_values_normalised[0], clip_values_normalised[1])
        epsilon_normalised = epsilon / 255 * (clip_values_normalised[1] - clip_values_normalised[0])
        patch_normalised = (patch - classifier.preprocessing.mean) / classifier.preprocessing.std

        super().__init__(
            classifier,
            percent_poison,
            epsilon_normalised,
            max_trials,
            max_epochs,
            learning_rate_schedule,
            batch_size,
            clip_values_normalised,
            verbose,
        )
        self.indices_target = indices_target
        self.selection_strategy = selection_strategy
        self.patching_strategy = patching_strategy
        self.retraining_factor = retraining_factor
        self.model_retrain = model_retrain
        self.model_retraining_epoch = model_retraining_epoch
        self.indices_poison: np.ndarray
        self.patch = patch_normalised
        self.class_target = class_target
        self.class_source = class_source
        self.device_name = device_name

    # pylint: disable=W0221
    def poison(  # type: ignore
        self,
        x_trigger: np.ndarray,
        y_trigger: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes a portion of poisoned samples from x_train to make a model classify x_target
        as y_target by matching the gradients.

        :param x_trigger: A list of samples to use as triggers.
        :param y_trigger: A list of target classes to classify the triggers into.
        :param x_train: A list of training data to poison a portion of.
        :param y_train: A list of labels for x_train.
        :return: x_train, y_train and indices of poisoned samples.
                 Here, x_train are the samples selected from target class
                 in training data.
        """
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification import TensorFlowV2Classifier

        # Apply Normalisation
        x_train = np.copy(x_train)
        x_trigger = (
            x_trigger - self.substitute_classifier.preprocessing.mean
        ) / self.substitute_classifier.preprocessing.std
        x_train = (
            x_train - self.substitute_classifier.preprocessing.mean
        ) / self.substitute_classifier.preprocessing.std

        x_train_target_samples, y_train_target_samples = self._select_target_train_samples(x_train, y_train)
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            poisoner = self._poison__pytorch
            finish_poisoning = self._finish_poison_pytorch
            initializer = self._initialize_poison_pytorch
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            poisoner = self._poison__tensorflow
            finish_poisoning = self._finish_poison_tensorflow
            initializer = self._initialize_poison_tensorflow
        else:
            raise NotImplementedError("SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.")

        # Choose samples to poison.
        x_trigger = self._apply_trigger_patch(x_trigger)
        if len(np.shape(y_trigger)) == 2:  # dense labels
            classes_target = set(np.argmax(y_trigger, axis=-1))
        else:  # sparse labels
            classes_target = set(y_trigger)
        num_poison_samples = int(self.percent_poison * len(x_train_target_samples))

        # Try poisoning num_trials times and choose the best one.
        best_B = np.finfo(np.float32).max  # pylint: disable=C0103
        best_x_poisoned: np.ndarray
        best_indices_poison: np.ndarray

        if len(np.shape(y_train)) == 2:
            y_train_classes = np.argmax(y_train_target_samples, axis=-1)
        else:
            y_train_classes = y_train_target_samples
        for _ in trange(self.max_trials):
            if self.selection_strategy == "random":
                self.indices_poison = np.random.permutation(
                    np.where([y in classes_target for y in y_train_classes])[0]
                )[:num_poison_samples]
            else:
                self.indices_poison = self._select_poison_indices(
                    self.substitute_classifier, x_train_target_samples, y_train_target_samples, num_poison_samples
                )
            x_poison = x_train_target_samples[self.indices_poison]
            y_poison = y_train_target_samples[self.indices_poison]
            initializer(x_trigger, y_trigger, x_poison, y_poison)
            original_epochs = self.max_epochs
            if self.model_retrain:
                retrain_epochs = self.max_epochs // self.retraining_factor
                for i in range(self.retraining_factor):
                    if i == self.retraining_factor - 1:
                        self.max_epochs = original_epochs - retrain_epochs * i
                        x_poisoned, B_ = poisoner(x_poison, y_poison)  # pylint: disable=C0103
                    else:
                        self.max_epochs = retrain_epochs
                        x_poisoned, B_ = poisoner(x_poison, y_poison)  # pylint: disable=C0103
                        self._model_retraining(x_poisoned, x_train, y_train, x_test, y_test)
            else:
                x_poisoned, B_ = poisoner(x_poison, y_poison)  # pylint: disable=C0103
            finish_poisoning()
            B_ = np.mean(B_)  # Averaging B losses from multiple batches.  # pylint: disable=C0103
            if B_ < best_B:
                best_B = B_  # pylint: disable=C0103
                best_x_poisoned = x_poisoned
                best_indices_poison = self.indices_poison

        # Apply De-Normalization
        x_train = x_train * self.substitute_classifier.preprocessing.std + self.substitute_classifier.preprocessing.mean
        best_x_poisoned = (
            best_x_poisoned * self.substitute_classifier.preprocessing.std
            + self.substitute_classifier.preprocessing.mean
        )
        if self.verbose > 0:
            logger.info("Best B-score: %s", best_B)
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            x_train[self.indices_target[best_indices_poison]] = best_x_poisoned
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            x_train[self.indices_target[best_indices_poison]] = best_x_poisoned
        else:
            raise NotImplementedError("SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.")
        return x_train, y_train

    def _select_target_train_samples(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Used for selecting train samples from target class
        :param x_train: clean training data
        :param y_train: labels fo clean training data
        :return x_train_target_samples, y_train_target_samples:
        samples and labels selected from target class in train data
        """
        x_train_samples = np.copy(x_train)
        index_target = np.where(y_train.argmax(axis=1) == self.class_target)[0]
        x_train_target_samples = x_train_samples[index_target]
        y_train_target_samples = y_train[index_target]
        return x_train_target_samples, y_train_target_samples

    def get_poison_indices(self) -> np.ndarray:
        """
        :return: indices of best poison index
        """
        return self.indices_poison

    def _model_retraining(
        self,
        poisoned_samples: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Applies retraining to substitute model

        :param poisoned_samples: poisoned array.
        :param x_train: clean training data.
        :param y_train: labels for training data.
        :param x_test: clean test data.
        :param y_test: labels for test data.
        """
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification import TensorFlowV2Classifier

        model_: Union[TensorFlowV2Classifier, PyTorchClassifier]
        x_train_un = np.copy(x_train)
        x_train_un[self.indices_target[self.indices_poison]] = poisoned_samples
        x_train_un = x_train_un * self.substitute_classifier.preprocessing.std
        x_train_un += self.substitute_classifier.preprocessing.mean
        num_classes = self.substitute_classifier.nb_classes

        if isinstance(self.substitute_classifier, PyTorchClassifier):
            model_ = self._create_model(
                x_train_un,
                y_train,
                x_test=x_test,
                y_test=y_test,
                num_classes=num_classes,
                batch_size=128,
                epochs=self.model_retraining_epoch,
            )
            check_train = self.substitute_classifier.model.training
            self.substitute_classifier = model_
            self.substitute_classifier.model.training = check_train

        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            model_ = self._create_model(
                x_train_un,
                y_train,
                x_test=x_test,
                y_test=y_test,
                num_classes=num_classes,
                batch_size=128,
                epochs=self.model_retraining_epoch,
            )
            check_train = self.substitute_classifier.model.trainable
            self.substitute_classifier = model_
            self.substitute_classifier.model.trainable = check_train

        else:
            raise NotImplementedError("SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.")

    def _create_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        num_classes: int = 10,
        batch_size: int = 128,
        epochs: int = 80,
    ) -> Tuple[Any, Any, Any]:
        """
        Creates a new model.

        :param x_train: Samples of train data.
        :param y_train: Labels of train data.
        :param x_test: Samples of test data.
        :param y_test: Labels of test data.
        :param num_classes: Number of classes of labels in train data.
        :param batch_size: The size of batch used for training.
        :param epochs: The number of epochs for which training need to be applied.
        :return model, loss_fn, optimizer - trained model, loss function used to train the model and optimizer used.
        """
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification import TensorFlowV2Classifier

        if isinstance(self.substitute_classifier, PyTorchClassifier):
            # Reset Weights of the newly initialized model
            model_new = self.substitute_classifier.clone_for_refitting()
            for layer in model_new.model.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
            model_new.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs, verbose=1)
            
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            model_new = self.substitute_classifier.clone_for_refitting()
            model_new.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs, verbose=1)

        else:
            raise ValueError("SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.")
        return model_new

    # This function is responsible for returning indices of poison images with maximum gradient norm
    def _select_poison_indices(
        self, classifier: "CLASSIFIER_NEURALNETWORK_TYPE", x_samples: np.ndarray, y_samples: np.ndarray, num_poison: int
    ) -> np.ndarray:
        """
        Select indices of poisoned samples

        :classifier: Substitute Model.
        :x_samples: Samples of poison. [x_samples are normalised]
        :y_samples: Labels of samples of poison.
        :num_poison: Number of poisoned samples to be selected out of all x_samples.
        :return indices - Indices of samples to be poisoned.
        """
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification import TensorFlowV2Classifier

        if isinstance(self.substitute_classifier, PyTorchClassifier):
            import torch

            device = torch.device(self.device_name)
            grad_norms = []
            criterion = torch.nn.CrossEntropyLoss()
            model = classifier.model
            model.eval()
            differentiable_params = [p for p in classifier.model.parameters() if p.requires_grad]
            for x, y in zip(x_samples, y_samples):
                image = torch.tensor(x, dtype=torch.float32).float().to(device)
                label = torch.tensor(y).to(device)
                loss_pt = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = list(torch.autograd.grad(loss_pt, differentiable_params, only_inputs=True))
                grad_norm = torch.tensor(0, dtype=torch.float32).to(device)
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt())
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            import tensorflow as tf

            grad_norms = []
            for i in range(len(x_samples) - 1):
                image = tf.constant(x_samples[i : i + 1])
                label = tf.constant(y_samples[i : i + 1])
                with tf.GradientTape() as t:  # pylint: disable=C0103
                    t.watch(classifier.model.weights)
                    output = classifier.model(image, training=False)
                    loss_tf = classifier.model.compiled_loss(label, output)  # type: ignore
                    gradients = list(t.gradient(loss_tf, classifier.model.weights))
                    gradients = [w for w in gradients if w is not None]
                    grad_norm = tf.constant(0, dtype=tf.float32)
                    for grad in gradients:
                        grad_norm += tf.reduce_sum(tf.math.square(grad))
                    grad_norms.append(tf.math.sqrt(grad_norm))
        else:
            raise NotImplementedError("SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.")
        indices = sorted(range(len(grad_norms)), key=lambda k: grad_norms[k])
        indices = indices[-num_poison:]
        return np.array(indices)  # this will get only indices for target class

    # This function is responsible for applying trigger patches to the images
    # fixed - where the trigger is applied at the bottom right of the image
    # random - where the trigger is applied at random location of the image
    def _apply_trigger_patch(self, x_trigger: np.ndarray) -> np.ndarray:
        """
        Select indices of poisoned samples

        :x_trigger: Samples to be used for trigger.
        :return tensor with applied trigger patches.
        """
        patch_size = self.patch.shape[1]
        if self.patching_strategy == "fixed":
            if self.estimator.channels_first:
                x_trigger[:, :, -patch_size:, -patch_size:] = self.patch
            else:
                x_trigger[:, -patch_size:, -patch_size:, :] = self.patch
        else:
            for x in x_trigger:
                if self.estimator.channels_first:
                    x_cord = random.randrange(0, x.shape[1] - self.patch.shape[1] + 1)
                    y_cord = random.randrange(0, x.shape[2] - self.patch.shape[2] + 1)
                    x[:, x_cord : x_cord + patch_size, y_cord : y_cord + patch_size] = self.patch
                else:
                    x_cord = random.randrange(0, x.shape[0] - self.patch.shape[0] + 1)
                    y_cord = random.randrange(0, x.shape[1] - self.patch.shape[1] + 1)
                    x[x_cord : x_cord + patch_size, y_cord : y_cord + patch_size, :] = self.patch

        return x_trigger
