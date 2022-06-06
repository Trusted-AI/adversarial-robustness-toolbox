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
This module implements Sleeper Agent clean-label attacks on Neural Networks.

| Paper link: https://arxiv.org/abs/2106.08970
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, Tuple, TYPE_CHECKING, List

import numpy as np
from tqdm.auto import trange, tqdm
import random

from art.attacks.attack import Attack
from art.attacks.poisoning import GradientMatchingAttack
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)
import pdb


class SleeperAgentAttack(GradientMatchingAttack):
    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        percent_poison: float,
        epsilon: float = 0.1,
        max_trials: int = 8,
        max_epochs: int = 250,
        learning_rate_schedule: Tuple[List[float], List[int]] = ([1e-1, 1e-2, 1e-3, 1e-4], [100, 150, 200, 220]),
        batch_size: int = 128,
        clip_values: Tuple[float, float] = (0, 1.0),
        verbose: int = 1,
        indices_target = None,
        patching_strategy = "random",
        selection_strategy = "random",  
        retraining_factor = 1,
        model_retraining = False,
        model_retraining_epoch = 1,
        patch = None
    ):
        super().__init__(classifier,
                         percent_poison,
                         epsilon,
                         max_trials,
                         max_epochs,
                         learning_rate_schedule,
                         batch_size,
                         clip_values,
                         verbose)
        self.indices_target = indices_target
        self.selection_strategy = selection_strategy
        self.patching_strategy = patching_strategy
        self.retraining_factor = retraining_factor
        self.model_retraining = model_retraining
        self.model_retraining_epoch = model_retraining_epoch
        self.indices_poison = []
        self.patch = patch
        

    """
    Implementation of Sleeper Agent Attack"""
    def poison(
        self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes a portion of poisoned samples from x_train to make a model classify x_target
        as y_target by matching the gradients.

        :param x_trigger: A list of samples to use as triggers.
        :param y_trigger: A list of target classes to classify the triggers into.
        :param x_train: A list of training data to poison a portion of.
        :param y_train: A list of labels for x_train.
        :return: A list of poisoned samples, and y_train.
        """
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification.tensorflow import TensorFlowV2Classifier

        if isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            poisoner = self._GradientMatchingAttack__poison__tensorflow
            finish_poisoning = self._GradientMatchingAttack__finish_poison_tensorflow
        elif isinstance(self.substitute_classifier, PyTorchClassifier):
            poisoner = self._GradientMatchingAttack__poison__pytorch
            finish_poisoning = self._GradientMatchingAttack__finish_poison_pytorch
        else:
            raise NotImplementedError(
                "SleeperAgentAttack is currently implemented only for Tensorflow V2 and Pytorch."
            )
#         pdb.set_trace()
        # Choose samples to poison.
        x_train = np.copy(x_train)
        y_train = np.copy(y_train)
        x_trigger = self.apply_trigger_patch(x_trigger)
        if len(np.shape(y_trigger)) == 2:  # dense labels
            classes_target = set(np.argmax(y_trigger, axis=-1))
        else:  # sparse labels
            classes_target = set(y_trigger)
        num_poison_samples = int(self.percent_poison * len(x_train))

        # Try poisoning num_trials times and choose the best one.
        best_B = np.finfo(np.float32).max  # pylint: disable=C0103
        best_x_poisoned = None
        best_indices_poison = None

        if len(np.shape(y_train)) == 2:
            y_train_classes = np.argmax(y_train, axis=-1)
        else:
            y_train_classes = y_train
        for _ in trange(self.max_trials):
            if self.selection_strategy == "random":
                self.indices_poison = np.random.permutation(np.where([y in classes_target for y in y_train_classes])[0])[:num_poison_samples]
            else:
                self.indices_poison = self.select_poison_indices(self.substitute_classifier,x_train,y_train,num_poison_samples)    
            x_poison = x_train[self.indices_poison]
            y_poison = y_train[self.indices_poison]
            self._GradientMatchingAttack__initialize_poison(x_trigger, y_trigger, x_poison, y_poison)
            if self.model_retraining:
                retrain_epochs = self.retraining_factor//self.max_epochs
                for i in range(self.retraining_factor-1):
                    self.max_epochs = retrain_epochs 
                    x_poisoned, B_ = poisoner(x_poison, y_poison) 
                    self.model_retraining(x_poisoned)
            else:
                x_poisoned, B_ = poisoner(x_poison, y_poison)   # pylint: disable=C0103
            finish_poisoning()
            B_ = np.mean(B_)  # Averaging B losses from multiple batches.  # pylint: disable=C0103
            if B_ < best_B:
                best_B = B_  # pylint: disable=C0103
                best_x_poisoned = x_poisoned
                best_indices_poison = self.indices_poison

        if self.verbose > 0:
            print("Best B-score:", best_B)
        x_train[best_indices_poison] = best_x_poisoned
        return x_train, y_train, best_indices_poison 
    
    def model_retraining(self,poisoned_samples):
        import torch
        from art.utils import load_cifar10
        (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
        mean = np.mean(x_train,axis=(0,1,2,3))
        std = np.std(x_train,axis=(0,1,2,3))
        x_train = (x_train-mean)/(std+1e-7)
        x_test = (x_test-mean)/(std+1e-7)
        min_ = (min_-mean)/(std+1e-7)
        max_ = (max_-mean)/(std+1e-7)
        x_train = np.transpose(x_train, [0, 3,1,2])
        poisoned_samples = np.asarray(poisoned_samples)
        x_train[self.indices_target[self.indices_poison]] = poisoned_samples
        model,loss_fn,optimizer = create_model(x_train, y_train, x_test=x_test, y_test=y_test,
                                               num_classes=10, batch_size=128, epochs=self.model_retraining_epoch)
        model_ = PyTorchClassifier(model, input_shape=x_train.shape[1:], loss=loss_fn,
                                   optimizer=optimizer, nb_classes=10)
        check_train = self.substitute_classifier.model.training 
        self.substitute_classifier = model_
        self.substitute_classifier.model.training = check_train
        
    def create_model(x_train, y_train, x_test=None, y_test=None, num_classes=10, batch_size=128,
                     epochs=80):
        from torchvision.models.resnet import BasicBlock, Bottleneck
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        import torchvision
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2],
                                          num_classes=num_classes)
        # Define the loss function with Classification Cross-Entropy loss and an 
        # optimizer with Adam optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True)
        model.to(device)
        y_train = np.argmax(y_train, axis=1)
        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device) # transform to torch tensor
        y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        x_test = np.transpose(x_test, [0, 3,1,2])
        y_test = np.argmax(y_test, axis=1)
        x_tensor_test = torch.tensor(x_test, dtype=torch.float32, device=device) # transform to torch tensor
        y_tensor_test = torch.tensor(y_test, dtype=torch.long, device=device)

        dataset_train = TensorDataset(x_tensor,y_tensor) # create your datset
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

        dataset_test = TensorDataset(x_tensor_test,y_tensor_test) # create your datset
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

        for epoch in trange(epochs):
            running_loss = 0.0
            total = 0
            accuracy = 0
        for i, data in enumerate(dataloader_train, 0):
            inputs, labels = data
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
        train_accuracy = (100 * accuracy / total)
        print("Epoch %d train accuracy: %f" % (epoch, train_accuracy))
        test_accuracy = testAccuracy(model, dataloader_test)
        print("Final test accuracy: %f" % test_accuracy)
        return model,loss_fn,optimizer


    def select_poison_indices(self,classifier,x_samples,y_samples,num_poison):
        # CHECK IF THE MODEL IS TRAIN/EVAL?????
        # pdb.set_trace()
        import torch
        # Here, x are the samples for target class only then we select from all those      
        device = "cuda" if torch.cuda.is_available() else "cpu"

        grad_norms = []
        criterion = torch.nn.CrossEntropyLoss()
        model = classifier.model
        model.eval()
        differentiable_params = [p for p in classifier.model.parameters() if p.requires_grad]
        for x,y in zip(x_samples,y_samples):
            image = torch.tensor(x).to(device)
            label = torch.tensor(y).to(device)
            loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            grad_norms.append(grad_norm.sqrt())      

        indices = sorted(range(len(grad_norms)), key=lambda k: grad_norms[k])
        indices = indices[-num_poison:]
        return indices # this will get only indices for target class
    
    def apply_trigger_patch(self,x_trigger):    
        from art.estimators.classification.pytorch import PyTorchClassifier
        if self.patching_strategy == "fixed":
            x_trigger[:,-8:,-8:,:] = self.patch
        else:
            for x in x_trigger:
                x_cord = random.randrange(0,x.shape[1] - self.patch.shape[1] + 1)
                y_cord = random.randrange(0,x.shape[2] - self.patch.shape[2] + 1)
                x[x_cord:x_cord+8,y_cord:y_cord+8,:]= self.patch
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            import torch
            
            return torch.tensor(np.transpose(x_trigger, [0, 3,1,2]))