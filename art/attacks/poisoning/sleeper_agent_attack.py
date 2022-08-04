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
from typing import Any, Tuple, TYPE_CHECKING, List
import random

import numpy as np
from tqdm.auto import trange


from art.attacks.poisoning import GradientMatchingAttack

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
    import torch

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
        super().__init__(
            classifier,
            percent_poison,
            epsilon,
            max_trials,
            max_epochs,
            learning_rate_schedule,
            batch_size,
            clip_values,
            verbose,
        )
        self.indices_target = indices_target
        self.selection_strategy = selection_strategy
        self.patching_strategy = patching_strategy
        self.retraining_factor = retraining_factor
        self.model_retrain = model_retrain
        self.model_retraining_epoch = model_retraining_epoch
        self.indices_poison: np.ndarray
        self.patch = patch
        self.class_target = class_target
        self.class_source = class_source

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

        x_train_target_samples, y_train_target_samples = self.select_target_train_samples(x_train, y_train)
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            poisoner = self._poison__pytorch
            finish_poisoning = self._finish_poison_pytorch
            initializer = self._initialize_poison_pytorch
            if self.estimator.channels_first:
                x_train_target_samples = np.transpose(x_train_target_samples, [0, 3, 1, 2])
        else:
            raise NotImplementedError("SleeperAgentAttack is currently implemented only for PyTorch.")

        # Choose samples to poison.
        x_trigger = self.apply_trigger_patch(x_trigger)
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
                self.indices_poison = self.select_poison_indices(
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
                        self.model_retraining(x_poisoned, x_train, y_train, x_test, y_test)
            else:
                x_poisoned, B_ = poisoner(x_poison, y_poison)  # pylint: disable=C0103
            finish_poisoning()
            B_ = np.mean(B_)  # Averaging B losses from multiple batches.  # pylint: disable=C0103
            if B_ < best_B:
                best_B = B_  # pylint: disable=C0103
                best_x_poisoned = x_poisoned
                best_indices_poison = self.indices_poison

        if self.verbose > 0:
            print("Best B-score:", best_B)
        x_train[self.indices_target[best_indices_poison]] = np.transpose(best_x_poisoned, [0, 2, 3, 1])
        return x_train, y_train

    def select_target_train_samples(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def model_retraining(
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

        x_train = np.transpose(x_train, [0, 3, 1, 2])

        x_train[self.indices_target[self.indices_poison]] = poisoned_samples
        model, loss_fn, optimizer = self.create_model(
            x_train,
            y_train,
            x_test=x_test,
            y_test=y_test,
            num_classes=10,
            batch_size=128,
            epochs=self.model_retraining_epoch,
        )
        model_ = PyTorchClassifier(
            model, input_shape=x_train.shape[1:], loss=loss_fn, optimizer=optimizer, nb_classes=10
        )
        check_train = self.substitute_classifier.model.training
        self.substitute_classifier = model_
        self.substitute_classifier.model.training = check_train

    def create_model(
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
        import torch  # lgtm [py/repeated-import]
        import torchvision

        device = self.estimator.device
        model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        model.to(device)
        y_train = np.argmax(y_train, axis=1)
        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)  # transform to torch tensor
        y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        x_test = np.transpose(x_test, [0, 3, 1, 2])
        y_test = np.argmax(y_test, axis=1)
        x_tensor_test = torch.tensor(x_test, dtype=torch.float32, device=device)  # transform to torch tensor
        y_tensor_test = torch.tensor(y_test, dtype=torch.long, device=device)

        dataset_train = torch.utils.data.TensorDataset(x_tensor, y_tensor)  # create your dataset
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)

        dataset_test = torch.utils.data.TensorDataset(x_tensor_test, y_tensor_test)  # create your dataset
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

        for epoch in trange(epochs):
            running_loss = 0.0
            total = 0
            accuracy = 0
            for _, data in enumerate(dataloader_train, 0):
                inputs, labels = data
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
                running_loss += loss.item()
            if (epoch % 5 == 0) or epoch == (epochs - 1):
                train_accuracy = 100 * accuracy / total
                logger.info("Epoch %d train accuracy: %f", epoch, train_accuracy)
        test_accuracy = self.test_accuracy(model, dataloader_test)
        logger.info("Final test accuracy: %f", test_accuracy)
        return model, loss_fn, optimizer

    @classmethod
    def test_accuracy(cls, model: "torch.nn.Module", test_loader: "torch.utils.data.dataloader.DataLoader") -> float:
        """
        Calculates test accuracy on trained model

        :param model: Trained model.
        :return accuracy - accuracy of trained model on test data.
        """
        import torch  # lgtm [py/repeated-import]

        model_was_training = model.training
        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # run the model on the test set to predict labels
                outputs = model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = 100 * accuracy / total
        if model_was_training:
            model.train()
        return accuracy

    # This function is responsible for returning indices of poison images with maximum gradient norm
    @classmethod
    def select_poison_indices(
        cls, classifier: "CLASSIFIER_NEURALNETWORK_TYPE", x_samples: np.ndarray, y_samples: np.ndarray, num_poison: int
    ) -> np.ndarray:
        """
        Select indices of poisoned samples

        :classifier: Substitute Model.
        :x_samples: Samples of poison.
        :y_samples: Labels of samples of poison.
        :num_poison: Number of poisoned samples to be selected out of all x_samples.
        :return indices - Indices of samples to be poisoned.
        """
        import torch  # lgtm [py/repeated-import]
        from art.estimators.classification.pytorch import PyTorchClassifier

        if isinstance(classifier, PyTorchClassifier):
            device = classifier.device
        else:
            raise ValueError("Classifier is not of type PyTorchClassifier.")
        grad_norms = []
        criterion = torch.nn.CrossEntropyLoss()
        model = classifier.model
        model.eval()
        differentiable_params = [p for p in classifier.model.parameters() if p.requires_grad]
        for x, y in zip(x_samples, y_samples):
            image = torch.tensor(x, dtype=torch.float32).float().to(device)
            label = torch.tensor(y).to(device)
            loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
            grad_norm = torch.tensor(0, dtype=torch.float32).to(device)
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            grad_norms.append(grad_norm.sqrt())

        indices = sorted(range(len(grad_norms)), key=lambda k: grad_norms[k])
        indices = indices[-num_poison:]
        return np.array(indices)  # this will get only indices for target class

    # This function is responsible for applying trigger patches to the images
    # fixed - where the trigger is applied at the bottom right of the image
    # random - where the trigger is applied at random location of the image
    def apply_trigger_patch(self, x_trigger: np.ndarray) -> np.ndarray:
        """
        Select indices of poisoned samples

        :x_trigger: Samples to be used for trigger.
        :return tensor with applied trigger patches.
        """
        patch_size = self.patch.shape[1]
        if self.patching_strategy == "fixed":
            x_trigger[:, -patch_size:, -patch_size:, :] = self.patch
        else:
            for x in x_trigger:
                x_cord = random.randrange(0, x.shape[1] - self.patch.shape[1] + 1)
                y_cord = random.randrange(0, x.shape[2] - self.patch.shape[2] + 1)
                x[x_cord : x_cord + patch_size, y_cord : y_cord + patch_size, :] = self.patch

        if self.estimator.channels_first:
            return np.transpose(x_trigger, [0, 3, 1, 2])

        return x_trigger
