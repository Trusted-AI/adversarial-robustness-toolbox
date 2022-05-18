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
This module implements Smooth Adversarial Attack using PGD and DDN.

| Paper link: https://arxiv.org/pdf/1906.04584.pdf
| Authors' implementation: https://github.com/Hadisalman/smoothing-adversarial
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from art.config import ART_NUMPY_DTYPE
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)

def fit_pytorch(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
    import torch.nn.functional as F
    from torch.distributions.normal import Normal
    import random
    import os
    from art.estimators.certification.randomized_smoothing.smooth_adversarial.smoothadvattack import Attacker, PGD_L2, DDN

    x = x.astype(ART_NUMPY_DTYPE)
    start_epoch = 0

    if self.attack_type == 'PGD':
        attacker = PGD_L2(steps=self.num_steps, device='cuda', max_norm=self.epsilon)
    elif self.attack_type == 'DDN':
        attacker = DDN(steps=self.num_steps, device='cuda', max_norm=self.epsilon)

    if self.optimizer is None:  # pragma: no cover
        raise ValueError("An optimizer is needed to train the model, but none for provided.")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError("A scheduler is needed to train the model, but none for provided.")
    if attacker is None:
        raise ValueError("A attacker is needed to smooth adversarially train the model, but none for provided.")

    num_batch = int(np.ceil(len(x) / float(batch_size)))
    ind = np.arange(len(x))

    # Start training
    for epoch_num in range(start_epoch+1, nb_epochs+1):
        # Shuffle the examples
        random.shuffle(ind)
        self.scheduler.step()
        # Put the model in the training mode
        self.model.train()
        self._requires_grad_(self.model, True)

        attacker.max_norm = np.min([self.epsilon, (epoch_num + 1) * self.epsilon/self.warmup])
        attacker.init_norm = np.min([self.epsilon, (epoch_num + 1) * self.epsilon/self.warmup])
        # Train for one epoch
        for nb in range(num_batch):
          i_batch = torch.from_numpy(x[ind[nb * batch_size : (nb + 1) * batch_size]]).to(self.device)
          o_batch = torch.from_numpy(y[ind[nb * batch_size : (nb + 1) * batch_size]]).to(self.device)

          mini_batches = self._get_minibatches(i_batch, o_batch, self.num_noise_vec)
          for inputs, targets in mini_batches:
            inputs = inputs.repeat((1, self.num_noise_vec, 1, 1)).view(i_batch.shape)
            noise = torch.randn_like(inputs, device=self.device) * self.scale

            #Attack and find adversarial examples
            self._requires_grad_(self.model, False)
            self.model.eval()
            inputs = attacker.attack(self.model, inputs, targets, 
                                    noise=noise, 
                                    num_noise_vectors=self.num_noise_vec, 
                                    no_grad=self.no_grad_attack,
                                    )
            self.model.train()
            self._requires_grad_(self.model, True)

            noisy_inputs = inputs + noise

            targets = targets.unsqueeze(1).repeat(1, self.num_noise_vec).reshape(-1,1).squeeze()
            outputs = self.model(noisy_inputs)
            loss = self.loss(outputs, targets)
                
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def get_batch_noisevec(X, num_noise_vec):
    batch_size = len(X)
    for i in range(num_noise_vec):
        yield X[i*batch_size : (i+1)*batch_size]

def fit_tensorflow(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
    import tensorflow as tf
    import torch.nn.functional as F
    import random
    import os
    from art.estimators.certification.randomized_smoothing.smooth_adversarial.smoothadvattack_tensorflow import Attacker, PGD_L2, DDN
    import pickle

    x = x.astype(ART_NUMPY_DTYPE)
    start_epoch = 0

    if self.attack_type == 'PGD':
        attacker = PGD_L2(steps=self.num_steps, device='cuda', max_norm=self.epsilon)
    elif self.attack_type == 'DDN':
        attacker = DDN(steps=self.num_steps, device='cuda', max_norm=self.epsilon)

    if self.optimizer is None:  # pragma: no cover
        raise ValueError("An optimizer is needed to train the model, but none for provided.")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError("A scheduler is needed to train the model, but none for provided.")
    if attacker is None:
        raise ValueError("A attacker is needed to smooth adversarially train the model, but none for provided.")

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = tf.transpose(x, (0,3,1,2))

    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size)

    # Start training
    for epoch_num in range(start_epoch+1, nb_epochs+1):
        attacker.max_norm = np.min([self.epsilon, (epoch_num + 1) * self.epsilon/self.warmup])
        attacker.init_norm = np.min([self.epsilon, (epoch_num + 1) * self.epsilon/self.warmup])
        for i_batch, o_batch in train_ds:
            mini_batches = get_minibatches(i_batch, o_batch, self.num_noise_vec)
            for inputs, targets in mini_batches:
                inputs = tf.reshape(
                            tf.tile(inputs, (1, self.num_noise_vec, 1, 1)),
                            i_batch.shape)
                noise = tf.random.normal(inputs.shape, 0, 1) * self.scale

                inputs = attacker.attack(self.model, inputs, targets, 
                                        noise=noise, 
                                        num_noise_vectors=self.num_noise_vec, 
                                        no_grad=self.no_grad_attack)

                noisy_inputs = inputs + noise
                noisy_inputs = tf.transpose(noisy_inputs, (0, 2, 3, 1))
                targets = tf.squeeze(
                            tf.reshape(
                              tf.tile(
                                tf.expand_dims(
                                  targets, axis=1), 
                                (1,self.num_noise_vec)
                              ), 
                              (-1,1)
                            )
                          )
                with tf.GradientTape() as tape:
                    predictions = self.model(noisy_inputs, training=True)
                    loss = self.loss_object(targets, predictions)

                self.optimizer.learning_rate = self.scheduler(epoch_num)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # End epoch

def _requires_grad_(self, model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)

def get_minibatches(X, y, num_batches):
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]