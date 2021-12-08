# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements Gradient Matching clean-label attacks (a.k.a. Witches' Brew) on Neural Networks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time
from typing import Optional, Tuple, Union, TYPE_CHECKING, List

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import PoisoningAttackWhiteBox
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.keras import KerasClassifier

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class GradientMatchingAttackKeras(PoisoningAttackWhiteBox):
    """
    Implementation of Gradient Matching Attack by Geiping, et. al. 2020.
    "Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching"

    | Paper link: https://arxiv.org/abs/2009.02276
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "target",
        # "feature_layer",
        "opt",
        "max_iter",
        "learning_rate",
        # "momentum",
        # "decay_iter",
        # "decay_coeff",
        "epsilon",
        "norm",
        # "dropout",
        # "endtoend",
        # "batch_size",
        "verbose",
    ]

    # _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin, KerasClassifier)
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier, # classifier: Union["CLASSIFIER_NEURALNETWORK_TYPE", List["CLASSIFIER_NEURALNETWORK_TYPE"]],  # TODO: Minimum requirement? classifier.model is a Tensorflow Layer.
        # target: np.ndarray,
        # feature_layer: Union[Union[str, int], List[Union[str, int]]],
        # opt: str = "adam",
        max_iter: int = 4000,
        learning_rate: float = 4e-2,
        # momentum: float = 0.9,
        # decay_iter: Union[int, List[int]] = 10000,
        # decay_coeff: float = 0.5,
        epsilon: float = 0.1,
        # dropout: float = 0.3,
        # net_repeat: int = 1,
        # endtoend: bool = True,
        batch_size: int = 128,
        verbose: bool = True,
    ):
        """
        Initialize an Feature Collision Clean-Label poisoning attack

        :param classifier: The proxy classifier used for the attack.
        :param target: The target input(s) of shape (N, W, H, C) to misclassify at test time. Multiple targets will be
                       averaged.
        # :param feature_layer: The name(s) of the feature representation layer(s).
        :param opt: The optimizer to use for the attack. Can be 'adam' or 'sgd'
        :param max_iter: The maximum number of iterations for the attack.
        :param learning_rate: The learning rate of clean-label attack optimization.
        # :param momentum: The momentum of clean-label attack optimization.
        # :param decay_iter: Which iterations to decay the learning rate.
        #                    Can be a integer (every N iterations) or list of integers [0, 500, 1500]
        # :param decay_coeff: The decay coefficient of the learning rate.
        :param epsilon: The perturbation budget
        # :param dropout: Dropout to apply while training
        # :param net_repeat: The number of times to repeat prediction on each network
        # :param endtoend: True for end-to-end training. False for transfer learning.
        # :param batch_size: Batch size.
        :param verbose: Show progress bars.
        """
        self.substitute_classifier = classifier

        super().__init__(classifier=self.substitute_classifier)  # type: ignore
        # self.target = target
        # self.opt = opt
        # self.momentum = momentum
        # self.decay_iter = decay_iter
        self.epsilon = epsilon
        # self.dropout = dropout
        # self.net_repeat = net_repeat
        # self.endtoend = endtoend
        # self.feature_layer = feature_layer
        self.learning_rate = learning_rate
        # self.decay_coeff = decay_coeff
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def poison(self, x_target: np.ndarray, y_target: np.ndarray, x_poison: np.ndarray, y_poison: np.ndarray, **kwargs) -> np.ndarray:
        """
        Iteratively finds optimal attack points starting at values at x

        :param x: The base images to begin the poison process.
        :param y: Target label
        :return: An tuple holding the (poisoning examples, poisoning labels).
        """
        # import torch  # lgtm [py/repeated-import]
        import tensorflow.keras.backend as K
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Embedding, Add

        # TODO 1: Choose the target sample to be misclassified.
        # TODO 2: Choose poison samples.
        # x_poison = [] # samples to be poisoned. Ideally of y_target class.
        # y_poison = [] # original y labels of the poison samples. This is a clean-label attack and it does not change the labels.
        # x_target = None # A single target sample to be misclassified.
        # y_target = None # A target class to classify x_target into.
        P = len(x_poison)

        # TODO 3: Get the target gradient vector.
        def grad_loss(model, input, target):
            with tf.GradientTape() as t:
                t.watch(model.weights)
                output = model(input)
                loss = model.compiled_loss(target, output)
            d_w = t.gradient(loss, model.trainable_weights)
            d_w = tf.concat([tf.reshape(d,[-1]) for d in d_w], 0)
            d_w_norm = d_w / tf.sqrt(tf.reduce_sum(tf.square(d_w)))
            return d_w_norm

        grad_ws_norm = grad_loss(self.substitute_classifier.model, tf.constant(x_target), tf.constant(y_target))

        class ClipConstraint(tf.keras.constraints.MaxNorm):
            def __init__(self, max_value=2):
                super().__init__(max_value=max_value)

            def __call__(self, w):
                return tf.clip_by_value(w, -self.max_value, self.max_value)  # TODO: The poisoned sample needs to be normalized again.


        # TODO 4: Define the model to optimize.
        # input = model.input
        # input_target = Input(batch_shape=self.substitute_classifier.model.input.shape)
        input_poison = Input(batch_shape=self.substitute_classifier.model.input.shape)
        input_indices = Input(shape=())
        output = self.substitute_classifier.model.output
        # y_true_target = Input(batch_shape=output.shape)
        y_true_poison = Input(batch_shape=output.shape)
        # embedding_layer = Embedding(P, np.prod(input_poison.shape[1:]), embeddings_constraint=ClipConstraint(max_value=self.epsilon))  # REMARK: Tensorflow 2-2.7 has a bug not allowing constraints on sparse vectors.
        embedding_layer = Embedding(P, np.prod(input_poison.shape[1:]))
        embeddings = embedding_layer(input_indices)
        embeddings = ClipConstraint(max_value=self.epsilon)(embeddings)
        embeddings = tf.reshape(embeddings, tf.shape(input_poison))

        # class NoiseLayer(tf.keras.layers.Layer):
        #     def __init__(self):
        #         super(NoiseLayer, self).__init__()

        #     def build(self, input_shape):
        #         noise_shape = [P] + input_shape[1:]
        #         self.noise = self.add_weight("noise", shape=noise_shape)

        #     def call(self, inputs):
        #         return inputs + self.noise

        # noise_layer = NoiseLayer()
        # input_noised = noise_layer(input_poison)
        input_noised = Add()([input_poison, embeddings])
        # output_noised = self.substitute_classifier.model(input_noised)

        # class NormalizeNoise(tf.keras.callbacks.Callback):
        #     def on_train_batch_end(self, batch, logs=None):
        #         noise_layer.noise.assign(K.clip(noise_layer.noise, -self.epsilon, self.epsilon))  # TODO: The poisoned sample needs to be normalized again.

        def loss_fn(input_noised, target, grad_ws_norm):
            # with tf.GradientTape() as t:
            #     t.watch(self.substitute_classifier.model.trainable_weights)
            #     output2 = self.substitute_classifier.model(input_noised)
            #     loss2 = self.substitute_classifier.model.compiled_loss(target, output2)
            # d_w2 = t.gradient(loss2, self.substitute_classifier.model.trainable_weights)
            # d_w2 = tf.concat([tf.reshape(d,[-1]) for d in d_w2], 0)
            # d_w2_s = tf.sqrt(tf.sum(tf.square(d_w2)))
            # d_w2_norm = d_w2 / d_w2_s
            d_w2_norm = grad_loss(self.substitute_classifier.model, input_noised, target)
            B = 1 - tf.reduce_sum(grad_ws_norm * d_w2_norm)
            return B

        B = tf.keras.layers.Lambda(lambda x: loss_fn(x[0],x[1],x[2]))([input_noised, y_true_poison, grad_ws_norm])

        m = tf.keras.models.Model([input_poison, y_true_poison, input_indices], [input_noised, B])
        m.add_loss(B)

        model_trainable = self.substitute_classifier.model.trainable
        self.substitute_classifier.model.trainable = False

        class PredefinedLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, learning_rates, milestones):
                self.schedule = list(zip(milestones, learning_rates))
            def __call__(self, step):
                lr_prev = self.schedule[0][1]
                for m, lr in self.schedule:
                    if step < m:
                        return lr_prev
                    else:
                        lr_prev = lr
                return lr_prev

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(PredefinedLRSchedule([1e-1, 1e-2, 1e-3, 1e-4], [100, 150, 200, 220]))

        class SignedAdam(tf.keras.optimizers.Adam):
            def compute_gradients(self, loss, var_list=None,
                            gate_gradients=1,
                            aggregation_method=None,
                            colocate_gradients_with_ops=False,
                            grad_loss=None):
                grads_and_vars = super(SignedAdam, self).compute_gradients(loss, var_list,
                            gate_gradients,
                            aggregation_method,
                            colocate_gradients_with_ops,
                            grad_loss)
                return [(tf.sign(g),v) for (g,v) in grads_and_vars]

        m.compile(loss=None, optimizer=SignedAdam(learning_rate=0.1))

        self.substitute_classifier.model.trainable = model_trainable

        callbacks = [lr_schedule]  # NormalizeNoise()
        # Train the noise.
        m.fit([x_poison, y_poison, np.arange(len(y_poison))],
            callbacks=callbacks,
            batch_size=self.batch_size, epochs=250, verbose=self.verbose)

        # noise = noise_layer.noise.numpy()
        [input_noised_, B_] = m.predict([x_poison, y_poison, np.arange(len(y_poison))])
        return input_noised_

    def _check_params(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive")

        if self.max_iter < 1:
            raise ValueError("Value of max_iter at least 1")

        # if not isinstance(self.feature_layer, (str, int, list)):
        #     raise TypeError("Feature layer should be a string or int or list of string or int")

        # if self.opt.lower() not in ["adam", "sgd"]:
        #     raise ValueError("Optimizer must be 'adam' or 'sgd'")

        # if not 0 <= self.momentum <= 1:
        #     raise ValueError("Momentum must be between 0 and 1")

        # if isinstance(self.decay_iter, int) and self.decay_iter < 0:
        #     raise ValueError("decay_iter must be at least 0")

        # if isinstance(self.decay_iter, list) and not all(
        #     (isinstance(decay_iter, int) and decay_iter > 0 for decay_iter in self.decay_iter)
        # ):
            # raise ValueError("decay_iter is not a list of positive integers")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be at least 0")

        # if not 0 <= self.dropout <= 1:
        #     raise ValueError("dropout must be between 0 and 1")

        # if self.net_repeat < 1:
        #     raise ValueError("net_repeat must be at least 1")

        # if isinstance(self.feature_layer, list):
        #     for layer in self.feature_layer:
        #         if isinstance(layer, int):
        #             if not 0 <= layer < len(self.estimator.layer_names):
        #                 raise ValueError("feature_layer is not list of positive integers")
        #         elif not isinstance(layer, str):
        #             raise ValueError("feature_layer is not list of strings")

        # if isinstance(self.feature_layer, int):
        #     if not 0 <= self.feature_layer < len(self.estimator.layer_names):
        #         raise ValueError("feature_layer is not positive integer")

        # if not 0 <= self.decay_coeff <= 1:
        #     raise ValueError("Decay coefficient must be between zero and one")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")


# def get_poison_tuples(poison_batch, poison_label):
#     """
#     Includes the labels
#     """
#     poison = [
#         poison_batch.poison.data[num_p].unsqueeze(0).detach().cpu().numpy()
#         for num_p in range(poison_batch.poison.size(0))
#     ]
#     return np.vstack(poison), poison_label


# def loss_from_center(
#     subs_net_list, target_feat_list, poison_batch, net_repeat, end2end, feature_layer
# ) -> "torch.Tensor":
#     """
#     Calculate loss from center.
#     """
#     import torch  # lgtm [py/repeated-import]

#     if end2end:
#         loss = torch.tensor(0.0)
#         for net, center_feats in zip(subs_net_list, target_feat_list):
#             poisons_feats: Union[List[float], "torch.Tensor", np.ndarray]
#             if net_repeat > 1:
#                 poisons_feats_repeats = [
#                     net.get_activations(poison_batch(), layer=feature_layer, framework=True) for _ in range(net_repeat)
#                 ]
#                 block_num = len(poisons_feats_repeats[0])
#                 poisons_feats = []
#                 for block_idx in range(block_num):
#                     poisons_feats.append(
#                         sum([poisons_feat_r[block_idx] for poisons_feat_r in poisons_feats_repeats]) / net_repeat
#                     )
#             elif net_repeat == 1:
#                 if isinstance(feature_layer, list):
#                     poisons_feats = [
#                         torch.flatten(net.get_activations(poison_batch(), layer=layer, framework=True), 0)
#                         for layer in feature_layer
#                     ]
#                 else:  # pragma: no cover
#                     poisons_feats = net.get_activations(poison_batch(), layer=feature_layer, framework=True)
#             else:  # pragma: no cover
#                 assert False, "net_repeat set to {}".format(net_repeat)

#             net_loss = torch.tensor(0.0)
#             for pfeat, cfeat in zip(poisons_feats, center_feats):
#                 diff = torch.mean(pfeat, dim=0) - cfeat
#                 diff_norm = torch.norm(diff, dim=0)
#                 cfeat_norm = torch.norm(cfeat, dim=0)
#                 diff_norm = diff_norm / cfeat_norm
#                 net_loss += torch.mean(diff_norm)
#             loss += net_loss / len(center_feats)
#         loss = loss / len(subs_net_list)

#     else:  # pragma: no cover
#         loss = torch.tensor(0.0)
#         for net, center in zip(subs_net_list, target_feat_list):
#             poisons_list = [
#                 net.get_activations(poison_batch(), layer=feature_layer, framework=True) for _ in range(net_repeat)
#             ]
#             poisons = torch.tensor(sum(poisons_list) / len(poisons_list))

#             diff_2 = torch.mean(poisons, dim=0) - center
#             diff_norm = torch.norm(diff_2, dim=1) / torch.norm(center, dim=1)
#             loss += torch.mean(diff_norm)

#         loss = loss / len(subs_net_list)

#     return loss
