# MIT License
#
# Copyright (C) IBM Corporation 2018
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
| Paper link: https://openreview.net/forum?id=BJx040EFvH
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from art.classifiers import PyTorchClassifier

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.defences.trainer.trainer import Trainer
import apex.amp as amp
from art.utils import random_sphere
import time
import torch.nn.functional as F
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])
     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()

        return ({'input': x.float(), 'target': y.long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

DEVICE = 'gpu'


if DEVICE != 'cpu':
    trainset = torchvision.datasets.CIFAR10(root='/home/ambrish/github/cifar-data', train=True,
                                            download=True, transform=transform)
else:
    trainset = torchvision.datasets.CIFAR10(root='/Users/ambrish/github/cifar-data', train=True,
                                            download=True, transform=transform)

train_batches = Batches(trainset, 128, shuffle=True, set_random_choices=False, num_workers=2)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class AdversarialTrainerFBFPyTorch(Trainer):
    """
    | Paper link: https://openreview.net/forum?id=BJx040EFvH
    """

    def __init__(self, classifier, eps=8):
        """
        Create an :class:`.AdversarialTrainer` instance.

        :param classifier: Model to train adversarially.
        :type classifier: :class:`.Classifier`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`

        """
        self.eps = eps
        self.classifier = classifier
        if not isinstance(self.classifier,PyTorchClassifier):
            raise NotImplementedError
        # Setting up adversary and perform adversarial training:



    def fit(self, x, y, validation_data=None, batch_size=128, nb_epochs=20, **kwargs):
        """
        Train a model adversarially. See class documentation for more information on the exact procedure.

        :param x: Training set.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :type kwargs: `dict`
        :return: `None`
        """

        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        lr_schedule = lambda t: np.interp([t], [0, nb_epochs * 2 // 5, nb_epochs], [0, 0.21, 0])[0]

        for i_epoch in range(nb_epochs):
            logger.info("Adversarial training FBF epoch %i/%i", i_epoch, nb_epochs)

            # Shuffle the examples
            np.random.shuffle(ind)
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0

            # for batch_id in range(nb_batches):
            for batch_id, batch in enumerate(train_batches):
                X, y = batch['input'], batch['target']
                if DEVICE!='cpu':
                    x_batch = X.cpu().numpy()
                    y_batch = y.cpu().numpy()
                else:
                    x_batch = X.numpy()
                    y_batch = y.numpy()
                # lr = lr_schedule(i_epoch + (batch_id + 1) / nb_batches)
                #
                # self.classifier._optimizer.param_groups[0].update(lr=lr)
                # Create batch data
                # x_batch = x[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                # y_batch = y[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]]

                # x_batch_preprocessed, y_preprocessed = self.classifier._apply_preprocessing(x_batch, y_batch, fit=True)
                # #Raw pytorch FBF version
                # if self.classifier._reduce_labels:
                #     y_preprocessed = np.argmax(y_preprocessed, axis=1)
                #
                # i_batch = torch.from_numpy(x_batch_preprocessed).to(
                #     self.classifier._device)
                # o_batch = torch.from_numpy(y_preprocessed).to(
                #     self.classifier._device)
                #
                # cifar10_mean = (0.4914, 0.4822, 0.4465)
                # cifar10_std = (0.2471, 0.2435, 0.2616)
                # mu = torch.tensor(cifar10_mean).view(3, 1, 1).to(
                #     self.classifier._device)
                # std = torch.tensor(cifar10_std).view(3, 1, 1).to(
                #     self.classifier._device)
                # upper_limit = ((1 - mu) / std)
                # lower_limit = ((0 - mu) / std)
                #
                # epsilon = self.eps/std
                #
                # # n = x_batch.shape[0]
                # # m = np.prod(x_batch.shape[1:])
                # delta_rnd = np.random.uniform(low=-self.eps,high=self.eps,size=(x_batch.shape)).astype(ART_NUMPY_DTYPE)
                # # delta_rnd = random_sphere(n, m, self.eps, np.inf).reshape(x_batch.shape).astype(ART_NUMPY_DTYPE)
                # # delta_rnd = np.clip(delta_rnd, 0.0, 1.0)
                # delta_rnd_preprocessed,_ = self.classifier. _apply_preprocessing(delta_rnd, y=None, fit=False)
                # delta = torch.from_numpy(delta_rnd_preprocessed).to(
                #     self.classifier._device)
                # delta.requires_grad = True
                #
                #
                # # delta = torch.zeros_like(i_batch).to(
                # #     self.classifier._device)
                # # delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())
                # # delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(), epsilon[1][0][0].item())
                # # delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(), epsilon[2][0][0].item())
                # # delta.requires_grad = True
                # output = self.classifier._model(i_batch + delta)
                # # loss = self.classifier._loss(output[-1], o_batch)
                # loss = F.cross_entropy(output[-1], o_batch)
                # with amp.scale_loss(loss, self.classifier._optimizer) as scaled_loss:
                #     scaled_loss.backward()
                # # loss.backward()
                # # print(loss)
                # grad = delta.grad.detach()
                # delta.data = clamp(delta + 1.5 * epsilon * torch.sign(grad), -epsilon, epsilon)
                # delta = delta.detach()
                #
                # output = self.classifier._model(clamp(i_batch + delta[:i_batch.size(0)], lower_limit, upper_limit))
                # loss = self.classifier._loss(output[-1], o_batch)
                #
                # self.classifier._optimizer.zero_grad()
                # # loss.backward()
                # with amp.scale_loss(loss, self.classifier._optimizer) as scaled_loss:
                #     scaled_loss.backward()
                # nn.utils.clip_grad_norm_(self.classifier._model.parameters(), 0.5)
                # self.classifier._optimizer.step()

                n = x_batch.shape[0]
                m = np.prod(x_batch.shape[1:])
                delta = random_sphere(n, m, self.eps, np.inf).reshape(x_batch.shape).astype(ART_NUMPY_DTYPE)
                delta_grad = self.classifier.loss_gradient(x_batch + delta,y_batch)
                delta = np.clip(delta + 1.25*self.eps*np.sign(delta_grad), -self.eps, +self.eps)
                x_batch_pert = np.clip(x_batch+delta,self.classifier.clip_values[0], self.classifier.clip_values[1])

                # Fit batch
                self.classifier.fit(x_batch_pert, y_batch, nb_epochs=1, batch_size=x_batch.shape[0], **kwargs)

                # # Apply preprocessing
                # # x_preprocessed, y_preprocessed = self.classifier._apply_preprocessing(x_batch_pert, y_batch, fit=True)
                # x_preprocessed, _ = self.classifier._apply_preprocessing(x_batch_pert, None, fit=True)
                #
                # # # Check label shape
                # # if self.classifier._reduce_labels:
                # #     y_preprocessed = np.argmax(y_preprocessed, axis=1)
                #
                # i_batch = torch.from_numpy(x_preprocessed).to(
                #     self.classifier._device)
                # o_batch = torch.from_numpy(y_batch).to(
                #     self.classifier._device)
                #
                #
                # # Zero the parameter gradients
                # self.classifier._optimizer.zero_grad()
                #
                # # Perform prediction
                # model_outputs = self.classifier._model(i_batch)
                #
                # # Form the loss function
                # loss = self.classifier._loss(model_outputs[-1], o_batch)
                #
                # # Actual training
                # # loss.backward()
                # with amp.scale_loss(loss, self.classifier._optimizer) as scaled_loss:
                #     scaled_loss.backward()
                # # nn.utils.clip_grad_norm_(self.classifier._model.parameters(), 0.5)
                # self.classifier._optimizer.step()


                # train_loss += loss.item() * o_batch.size(0)
                # train_acc += (model_outputs[0].max(1)[1]==o_batch).sum().item()
                # train_n += o_batch.size(0)

            train_time = time.time()

            # compute accuracy
            (x_test, y_test) = validation_data
            output = np.argmax(self.predict(x_test), axis=1)
            nb_correct_pred = np.sum(output == np.argmax(y_test, axis=1))
            print('{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(i_epoch, train_time - start_time, 0.0,
                                                                        0.0, 0.0,
                                                                                 nb_correct_pred / x_test.shape[0]),
                  flush=True)
            # print('{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(i_epoch, train_time - start_time, lr,
            #                                                             train_loss / train_n, train_acc / train_n,
            #                                                                      nb_correct_pred / x_test.shape[0]),
            #       flush=True)

    def predict(self, x, **kwargs):
        """
        Perform prediction using the adversarially trained classifier.

        :param x: Test set.
        :type x: `np.ndarray`
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :type kwargs: `dict`
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)
