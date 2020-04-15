# MIT License
#
# Copyright (C) IBM Corporation 2020
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
This module implements the `Auto Projected Gradient Descent` attack.

| Paper link: https://arxiv.org/abs/2003.01690
"""
import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.attacks.evasion import FastGradientMethod
from art.utils import get_labels_np_array, check_and_transform_label_format, projection

logger = logging.getLogger(__name__)


class AutoProjectedGradientDescent(EvasionAttack):
    attack_params = FastGradientMethod.attack_params + ["max_iter", "random_eps"]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    _predefined_losses = [None, 'cross_entropy', 'difference_logits_ratio']

    def __init__(
            self,
            estimator,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False,
            batch_size=1,
            loss_type=None
    ):
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: An trained estimator.
        :type estimator: :class:`.BaseEstimator`
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
        if isinstance(estimator, TensorFlowV2Classifier):

            import tensorflow as tf
            from art.utils import is_probability

            if loss_type == 'cross_entropy':
                if is_probability(estimator.predict(x=np.ones(shape=(1, *estimator.input_shape)))):
                    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                else:
                    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            elif loss_type == 'difference_logits_ratio':
                if is_probability(estimator.predict(x=np.ones(shape=(1, *estimator.input_shape)))):
                    raise Exception
                else:
                    def custom_loss(y_true, y_pred):

                        # print('y_true')
                        # print(y_true)
                        # print('y_pred')
                        # print(y_pred)

                        i_y_true = tf.cast(tf.math.argmax(tf.cast(y_true, tf.int32), axis=1), tf.int32)

                        # print('i_y_true')
                        # print(i_y_true)

                        i_y_pred_arg = tf.argsort(y_pred, axis=1)

                        i_z_i_list = list()

                        for i in range(y_true.shape[0]):
                            if i_y_pred_arg[i, -1] != i_y_true[i]:
                                i_z_i_list.append(i_y_pred_arg[i, -1])
                            else:
                                i_z_i_list.append(i_y_pred_arg[i, -2])

                        i_z_i = tf.stack(i_z_i_list)

                        # print('i_z_i')
                        # print(i_z_i)

                        z_1 = tf.gather(y_pred, i_y_pred_arg[:, -1], axis=1, batch_dims=0)
                        z_3 = tf.gather(y_pred, i_y_pred_arg[:, -3], axis=1, batch_dims=0)
                        z_i = tf.gather(y_pred, i_z_i, axis=1, batch_dims=0)
                        z_y = tf.gather(y_pred, i_y_true, axis=1, batch_dims=0)

                        # print('--- gather')
                        # print('z_1')
                        # print(z_1)
                        # print('z_3')
                        # print(z_3)
                        # print('z_i')
                        # print(z_i)
                        # print('z_y')
                        # print(z_y)

                        z_1 = tf.linalg.diag_part(z_1)
                        z_3 = tf.linalg.diag_part(z_3)
                        z_i = tf.linalg.diag_part(z_i)
                        z_y = tf.linalg.diag_part(z_y)

                        # print('--- diag_part')
                        # print('z_1')
                        # print(z_1)
                        # print('z_3')
                        # print(z_3)
                        # print('z_i')
                        # print(z_i)
                        # print('z_y')
                        # print(z_y)

                        DLR = - (z_y - z_i) / (z_1 - z_3)

                        # print('DLR')
                        # print(DLR)

                        return DLR

                    loss_object = custom_loss
            elif loss_type is None:
                loss_object = estimator._loss_object
            else:
                raise Exception

            estimator_apgd = TensorFlowV2Classifier(model=estimator._model,
                                                    nb_classes=estimator.nb_classes,
                                                    input_shape=estimator.input_shape,
                                                    loss_object=loss_object,
                                                    train_step=estimator._train_step,
                                                    channel_index=estimator.channel_index,
                                                    clip_values=estimator.clip_values,
                                                    preprocessing_defences=estimator.preprocessing_defences,
                                                    postprocessing_defences=estimator.postprocessing_defences,
                                                    preprocessing=estimator.preprocessing,
                                                    )
        elif isinstance(estimator, PyTorchClassifier):
            pass
        else:
            raise Exception

        # super().__init__(
        #     estimator=estimator_apgd,
        #     norm=norm,
        #     eps=eps,
        #     eps_step=eps_step,
        #     targeted=targeted,
        #     num_random_init=0,
        #     batch_size=batch_size,
        #     minimal=False,
        # )

        super().__init__(estimator=estimator_apgd)

        kwargs = {"max_iter": max_iter,
                  # "random_eps": False,
                  # "estimator": estimator_apgd,
                  "norm": norm,
                  "eps": eps,
                  "eps_step": eps_step,
                  "targeted": targeted,
                  # "num_random_init": 0,
                  "batch_size": batch_size,
                  # "minimal": False,
                  }
        self.set_params(**kwargs)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        mask = kwargs.get("mask")
        if mask is not None:
            # ensure the mask is broadcastable:
            if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape):]:
                raise ValueError("mask shape must be broadcastable to input shape")

        x_adv = x.astype(ART_NUMPY_DTYPE)

        for i_max_iter in range(self.max_iter):
            # x_adv = self._compute(
            #     x_adv,
            #     x,
            #     targets,
            #     mask,
            #     self.eps,
            #     self.eps_step,
            #     self._project,
            #     self.num_random_init > 0 and i_max_iter == 0,
            # )
            # def _compute(self, x, x_init, y, mask, eps, eps_step, project, random_init):

            # Compute perturbation with implicit batching
            for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):

                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                batch = x_adv[batch_index_1:batch_index_2]
                batch_labels = y[batch_index_1:batch_index_2]

                x_init = x[batch_index_1:batch_index_2]

                mask_batch = mask
                if mask is not None:
                    # Here we need to make a distinction: if the masks are different for each input, we need to index
                    # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                    if len(mask.shape) == len(x.shape):
                        mask_batch = mask[batch_index_1:batch_index_2]

                # Get perturbation
                # Pick a small scalar to avoid division by 0
                tol = 10e-8

                # Get gradient wrt loss; invert it if attack is targeted
                grad = self.estimator.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

                # Apply norm bound
                if self.norm == np.inf:
                    grad = np.sign(grad)
                elif self.norm == 1:
                    ind = tuple(range(1, len(batch.shape)))
                    grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
                elif self.norm == 2:
                    ind = tuple(range(1, len(batch.shape)))
                    grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
                assert batch.shape == grad.shape

                if mask_batch is None:
                    perturbation = grad
                else:
                    perturbation = grad * (mask_batch.astype(ART_NUMPY_DTYPE))

                # Apply perturbation and clip
                eps_step = self.eps_step

                batch = batch + eps_step * perturbation

                if self.estimator.clip_values is not None:
                    clip_min, clip_max = self.estimator.clip_values
                    batch = np.clip(batch, clip_min, clip_max)

                x_adv[batch_index_1:batch_index_2] = batch

                perturbation = projection(x_adv[batch_index_1:batch_index_2] - x_init, self.eps, self.norm)

                x_adv[batch_index_1:batch_index_2] = x_init + perturbation

        return x_adv

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
