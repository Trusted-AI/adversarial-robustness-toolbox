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

| Paper link: https://openreview.net/pdf?id=rJx1Na4Fwr
| Authors' implementation: https://github.com/RuntianZ/macer
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

from art.config import ART_NUMPY_DTYPE

logger = logging.getLogger(__name__)


def fit_pytorch_macer(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
    """
    Fit the randomized smoothed classifier for MACER training on the training set `(x, y)`.

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
    import torch.nn.functional as F
    from torch.distributions.normal import Normal
    import random

    x = x.astype(ART_NUMPY_DTYPE)
    m = Normal(torch.tensor([0.0]).to(self._device), torch.tensor([1.0]).to(self._device))  # pylint: disable=W0212
    cl_total = 0.0
    rl_total = 0.0
    input_total = 0
    start_epoch = 0

    # Put the model in the training mode
    self.model.train()

    if self.optimizer is None:  # pragma: no cover
        raise ValueError("An optimizer is needed to train the model, but none for provided.")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError("A scheduler is needed to train the model, but none for provided.")

    if kwargs.get("checkpoint") is not None:
        chkpt = kwargs.get("checkpoint")
        cpoint = torch.load(chkpt)
        self.model.load_state_dict(cpoint["net"])
        start_epoch = cpoint["epoch"]
        self.scheduler.step(start_epoch)
    num_batch = int(np.ceil(len(x) / float(batch_size)))
    ind = np.arange(len(x))

    # Start training
    for _ in range(start_epoch + 1, nb_epochs + 1):
        # Shuffle the examples
        random.shuffle(ind)
        i = 0
        # Train for one epoch
        for n_batch in range(num_batch):
            i_batch = torch.from_numpy(x[ind[n_batch * batch_size : (n_batch + 1) * batch_size]]).to(self.device)
            o_batch = torch.from_numpy(y[ind[n_batch * batch_size : (n_batch + 1) * batch_size]]).to(self.device)
            input_size = len(i_batch)
            input_total += input_size

            new_shape = [input_size * self.gauss_num]
            new_shape.extend(i_batch[0].shape)
            i_batch = i_batch.repeat((1, self.gauss_num, 1, 1)).view(new_shape)
            noise = torch.randn_like(i_batch, device=self.device) * self.scale
            noisy_inputs = i_batch + noise
            outputs = self.model(noisy_inputs)
            outputs = outputs.reshape((input_size, self.gauss_num, self.nb_classes))

            # Classification loss
            outputs_softmax = F.softmax(outputs, dim=2).mean(1)
            outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
            classification_loss = F.nll_loss(outputs_logsoftmax, o_batch, reduction="sum")

            cl_total += classification_loss.item()

            # Robustness loss
            beta_outputs = outputs * self.beta  # only apply beta to the robustness loss
            beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
            top2 = torch.topk(beta_outputs_softmax, 2)
            top2_score = top2[0]
            top2_idx = top2[1]
            indices_correct = top2_idx[:, 0] == o_batch  # G_theta
            out0, out1 = top2_score[indices_correct, 0], top2_score[indices_correct, 1]
            robustness_loss = m.icdf(out1) - m.icdf(out0)
            indices = (
                ~torch.isnan(robustness_loss)
                & ~torch.isinf(robustness_loss)
                & (torch.abs(robustness_loss) <= self.gamma)
            )  # hinge
            out0, out1 = out0[indices], out1[indices]
            robustness_loss = m.icdf(out1) - m.icdf(out0) + self.gamma
            robustness_loss = robustness_loss.sum() * self.scale / 2
            rl_total += robustness_loss.item()

            # Final objective function
            loss = classification_loss + self.lbd * robustness_loss
            loss /= input_size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i = i + 1

        self.scheduler.step()

        cl_total /= input_total
        rl_total /= input_total


def fit_tensorflow_macer(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int) -> None:
    """
    Fit the randomized smoothed classifier for MACER training on the training set `(x, y)` for tensorflow.

    :param x: Training data.
    :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
              (nb_samples,).
    :param batch_size: Batch size.
    :key nb_epochs: Number of epochs to use for training
    :param kwargs: Dictionary of framework-specific arguments.
    :return: `None`
    """
    import tensorflow as tf
    import math

    if self.optimizer is None:  # pragma: no cover
        raise ValueError("An optimizer is needed to train the model, but none for provided.")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError("A scheduler is needed to train the model, but none for provided.")

    loc_norm = tf.constant([0.0])
    scale_norm = tf.constant([1.0])
    cl_total = 0.0
    rl_total = 0.0
    input_total = 0
    start_epoch = 0
    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size)

    for epoch_num in range(start_epoch + 1, nb_epochs + 1):
        i = 0
        for images, labels in train_ds:
            images = tf.transpose(images, (0, 3, 1, 2))
            input_size = len(images)
            input_total += input_size
            new_shape = [input_size * self.gauss_num]
            new_shape.extend(images[0].shape)
            i_batch = tf.reshape(tf.tile(images, (1, self.gauss_num, 1, 1)), new_shape)
            i_batch = tf.transpose(i_batch, (0, 2, 3, 1))
            noise = tf.random.normal(i_batch.shape, 0, 1, tf.float32) * self.scale
            noisy_inputs = i_batch + noise
            with tf.GradientTape() as tape:
                outputs = self.model(noisy_inputs, training=True)
                outputs = tf.reshape(outputs, [input_size, self.gauss_num, self.nb_classes])
                # Classification loss
                outputs_softmax = tf.reduce_mean(tf.nn.softmax(outputs, axis=2), axis=1)
                outputs_logsoftmax = tf.math.log(outputs_softmax + 1e-10)
                indices = tf.stack([np.arange(len(labels)), labels], axis=1)
                nllloss = tf.gather_nd(outputs_logsoftmax, indices=indices)
                classification_loss = -tf.reduce_sum(nllloss)
                cl_total += tf.get_static_value(classification_loss)
                # Robustness loss
                beta_outputs = outputs * self.beta
                beta_outputs_softmax = tf.reduce_mean(tf.nn.softmax(beta_outputs, axis=2), axis=1)
                top2 = tf.math.top_k(beta_outputs_softmax, k=2)
                top2_score = top2[0]
                top2_idx = top2[1]
                indices_correct = top2_idx[:, 0] == labels
                out = tf.boolean_mask(top2_score, indices_correct)
                out0, out1 = out[:, 0], out[:, 1]
                icdf_out1 = loc_norm + scale_norm * tf.math.erfinv(2 * out1 - 1) * math.sqrt(2)
                icdf_out0 = loc_norm + scale_norm * tf.math.erfinv(2 * out0 - 1) * math.sqrt(2)
                robustness_loss = icdf_out1 - icdf_out0
                indices = (
                    ~tf.math.is_nan(robustness_loss)  # pylint: disable=E1130
                    & ~tf.math.is_inf(robustness_loss)  # pylint: disable=E1130
                    & (tf.abs(robustness_loss) <= self.gamma)
                )
                out0, out1 = out0[indices], out1[indices]
                icdf_out1 = loc_norm + scale_norm * tf.math.erfinv(2 * out1 - 1) * math.sqrt(2)
                icdf_out0 = loc_norm + scale_norm * tf.math.erfinv(2 * out0 - 1) * math.sqrt(2)
                robustness_loss = icdf_out1 - icdf_out0 + self.gamma
                robustness_loss = tf.reduce_sum(robustness_loss) * self.scale / 2
                rl_total += tf.get_static_value(robustness_loss)
                # Final objective function
                loss = classification_loss + self.lbd * robustness_loss
                loss /= input_size

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            i += 1

            self.optimizer.learning_rate = self.scheduler(epoch_num - 1)
            cl_total /= input_total
            rl_total /= input_total
