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
This module implements Smooth Adversarial Attack using PGD.
It supports only supports single noise version of PGD attack.

| Paper link: https://arxiv.org/pdf/1906.04584.pdf
| Authors' implementation: https://github.com/Hadisalman/smoothing-adversarial
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from art.config import ART_NUMPY_DTYPE

logger = logging.getLogger(__name__)


def fit_pytorch_smoothadv(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int) -> None:
    """
    Fit the randomized smoothed classifier for SmoothAdversarial training on the training set `(x, y)`
    in PyTorch.

    :param x: Training data.
    :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
              (nb_samples,).
    :param batch_size: Batch size.
    :key nb_epochs: Number of epochs to use for training
    :param kwargs: Dictionary of framework-specific arguments.
    :type kwargs: `dict`
    :return: `None`
    """
    import torch
    import random
    from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

    def requires_grad(model: torch.nn.Module, requires_grad_val: bool) -> None:
        """
        Sets the `requires_grad_` property for the model's parameters

        :param x: torch Model
        :param requires_grad_val: boolean value to set requires_grad_ property with.
        :return: `None`
        """
        for param in model.parameters():
            param.requires_grad_(requires_grad_val)

    x = x.astype(ART_NUMPY_DTYPE)
    start_epoch = 0
    attacker = None
    if self.attack_type == "PGD":
        attacker = ProjectedGradientDescent(self.estimator, eps=self.epsilon, max_iter=1, verbose=False)

    if self.optimizer is None:  # pragma: no cover
        raise ValueError("An optimizer is needed to train the model, but none for provided.")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError("A scheduler is needed to train the model, but none for provided.")
    if attacker is None:
        raise ValueError("A attacker is needed to smooth adversarially train the model, but none for provided.")

    num_batch = int(np.ceil(len(x) / float(batch_size)))
    ind = np.arange(len(x))

    # Start training
    for epoch_num in range(start_epoch + 1, nb_epochs + 1):
        # Shuffle the examples
        random.shuffle(ind)
        self.scheduler.step()
        # Put the model in the training mode
        self.model.train()
        requires_grad(self.model, True)

        attacker.norm = np.min([self.epsilon, (epoch_num + 1) * self.epsilon / self.warmup])
        # Train for one epoch
        for n_batch in range(num_batch):
            i_batch = torch.from_numpy(x[ind[n_batch * batch_size : (n_batch + 1) * batch_size]]).to(self.device)
            o_batch = torch.from_numpy(y[ind[n_batch * batch_size : (n_batch + 1) * batch_size]]).to(self.device)

            mini_batches = get_minibatches(i_batch, o_batch, self.num_noise_vec)
            for inputs, targets in mini_batches:
                inputs = inputs.repeat((1, self.num_noise_vec, 1, 1)).view(i_batch.shape)
                noise = torch.randn_like(inputs, device=self.device) * self.scale

                # Attack and find adversarial examples
                requires_grad(self.model, False)
                self.model.eval()

                original_inputs = inputs.cpu().detach().numpy()
                noise_for_attack = noise.cpu().detach().numpy()
                perturbation_delta = np.zeros_like(original_inputs)
                for _ in range(self.num_steps):
                    perturbed_inputs = original_inputs + perturbation_delta
                    adv_ex = attacker.generate(perturbed_inputs + noise_for_attack)
                    perturbation_delta = adv_ex - perturbed_inputs - noise_for_attack

                # update perturbated_inputs after last iteration
                perturbed_inputs = original_inputs + perturbation_delta
                self.model.train()
                requires_grad(self.model, True)
                noisy_inputs = torch.from_numpy(perturbed_inputs).to(self.device) + noise

                targets = targets.unsqueeze(1).repeat(1, self.num_noise_vec).reshape(-1, 1).squeeze()
                outputs = self.model(noisy_inputs)
                loss = self.loss(outputs, targets)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def get_batch_noisevec(x, num_noise_vec):
    """
    Yields the input tensor in batches iterating over number of noise vectors.

    :param x: Training data.
    :param num_noise_vec: Number of noise vector used for attack
    :return: `Iterable training data in batches`
    """
    batch_size = len(x)
    for i in range(num_noise_vec):
        yield x[i * batch_size : (i + 1) * batch_size]


def fit_tensorflow_smoothadv(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int) -> None:
    """
    Fit the randomized smoothed classifier for SmoothAdversarial training on the training set `(x, y)`
    in Tensorflow.

    :param x: Training data.
    :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
              (nb_samples,).
    :param batch_size: Batch size.
    :key nb_epochs: Number of epochs to use for training
    :param kwargs: Dictionary of framework-specific arguments.
    :type kwargs: `dict`
    :return: `None`
    """
    import tensorflow as tf
    from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

    x = x.astype(ART_NUMPY_DTYPE)
    start_epoch = 0
    attacker = None
    if self.attack_type == "PGD":
        attacker = ProjectedGradientDescent(self.estimator, eps=self.epsilon, max_iter=1, verbose=False)

    if self.optimizer is None:  # pragma: no cover
        raise ValueError("An optimizer is needed to train the model, but none for provided.")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError("A scheduler is needed to train the model, but none for provided.")
    if attacker is None:
        raise ValueError("A attacker is needed to smooth adversarially train the model, but none for provided.")

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size)

    # Start training
    for epoch_num in range(start_epoch + 1, nb_epochs + 1):
        attacker.norm = np.min([self.epsilon, (epoch_num + 1) * self.epsilon / self.warmup])
        for i_batch, o_batch in train_ds:
            mini_batches = get_minibatches(i_batch, o_batch, self.num_noise_vec)
            for inputs, targets in mini_batches:
                inputs = tf.reshape(tf.tile(inputs, (1, self.num_noise_vec, 1, 1)), i_batch.shape)
                noise = tf.random.normal(inputs.shape, 0, 1) * self.scale

                original_inputs = inputs.numpy()
                noise_for_attack = noise.numpy()
                perturbation_delta = np.zeros_like(original_inputs)
                for _ in range(self.num_steps):
                    perturbed_inputs = original_inputs + perturbation_delta
                    adv_ex = attacker.generate(perturbed_inputs + noise_for_attack)
                    perturbation_delta = adv_ex - perturbed_inputs - noise_for_attack

                # update perturbated_inputs after last iteration
                perturbed_inputs = original_inputs + perturbation_delta

                noisy_inputs = tf.convert_to_tensor(perturbed_inputs) + noise
                # noisy_inputs = tf.transpose(noisy_inputs, (0, 2, 3, 1))
                targets = tf.squeeze(
                    tf.reshape(tf.tile(tf.expand_dims(targets, axis=1), (1, self.num_noise_vec)), (-1, 1))
                )
                with tf.GradientTape() as tape:
                    predictions = self.model(noisy_inputs, training=True)
                    loss = self.loss_object(targets, predictions)

                self.optimizer.learning_rate = self.scheduler(epoch_num)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # End epoch


def get_minibatches(x, y, num_batches):
    """
    Yields the input tensor and target value pairs in batches.

    :param x: Training data.
    :param y: Target values.
    :param num_batches: Number of batches
    :return: `Iterable training data and target values pair in batches`
    """
    batch_size = len(x) // num_batches
    for i in range(num_batches):
        yield x[i * batch_size : (i + 1) * batch_size], y[i * batch_size : (i + 1) * batch_size]
