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
from typing import Tuple, TYPE_CHECKING, List

import numpy as np
from tqdm.auto import trange, tqdm

from art.attacks.attack import Attack
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class GradientMatchingAttack(Attack):
    """
    Implementation of Gradient Matching Attack by Geiping, et. al. 2020.
    "Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching"

    | Paper link: https://arxiv.org/abs/2009.02276
    """

    attack_params = Attack.attack_params + [
        "percent_poison",
        "max_trials",
        "max_epochs",
        "learning_rate_schedule",
        "epsilon",
        "clip_values",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

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
        verbose: bool = True,
    ):
        """
        Initialize a Gradient Matching Clean-Label poisoning attack (Witches' Brew).

        :param classifier: The proxy classifier used for the attack.
        :param percent_poison: The percentage of samples to poison among x_train.
        :param epsilon: The L-inf perturbation budget.
        :param max_trials: The maximum number of restarts to optimize the poison.
        :param max_epochs: The maximum number of epochs to optimize the train per trial.
        :param learning_rate_schedule: The learning rate schedule to optimize the poison.
            A List of (learning rate, epoch) pairs. The learning rate is used
            if the current epoch is less than the specified epoch.
        :param batch_size: Batch size.
        :param clip_values: The range of the input features to the classifier.
        :param verbose: Show progress bars.
        """
        self.substitute_classifier = classifier

        super().__init__(classifier)
        self.percent_poison = percent_poison
        self.epsilon = epsilon
        self.learning_rate_schedule = learning_rate_schedule
        self.max_trials = max_trials
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

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
            poisoner = self.__poison__tensorflow
        elif isinstance(self.substitute_classifier, PyTorchClassifier):
            poisoner = self.__poison__pytorch
        else:
            raise NotImplementedError(
                "GradientMatchingAttack is currently implemented only for Tensorflow V2 and Pytorch."
            )

        # Choose samples to poison.
        x_train = np.copy(x_train)
        y_train = np.copy(y_train)
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
            indices_poison = np.random.permutation(np.where([y in classes_target for y in y_train_classes])[0])[
                :num_poison_samples
            ]
            x_poison = x_train[indices_poison]
            y_poison = y_train[indices_poison]
            x_poisoned, B_ = poisoner(x_trigger, y_trigger, x_poison, y_poison)  # pylint: disable=C0103
            B_ = np.mean(B_)  # Averaging B losses from multiple batches.  # pylint: disable=C0103
            if B_ < best_B:
                best_B = B_  # pylint: disable=C0103
                best_x_poisoned = x_poisoned
                best_indices_poison = indices_poison

        x_train[best_indices_poison] = best_x_poisoned
        return x_train, y_train  # y_train has not been modified.

    def __poison__pytorch(
        self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_poison: np.ndarray, y_poison: np.ndarray
    ) -> np.ndarray:
        """
        Optimize the poison by matching the gradient within the perturbation budget.

        :param x_trigger: List of triggers.
        :param y_trigger: List of target labels.
        :param x_poison: List of samples to poison.
        :param y_poison: List of the labels for x_poison.
        :return: A pair of poisoned samples, B-score (cosine similarity of the gradients).
        """

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        num_poison = len(x_poison)
        len_noise = np.prod(x_poison.shape[1:])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def grad(classifier, x, target):
            classifier.model.zero_grad()
            y = classifier.model(x)
            loss_ = classifier.loss(y, target)
            loss_.backward()
            d_w = [w.grad for w in classifier.model.parameters()]
            d_w = torch.cat([w.flatten() for w in d_w])
            d_w_norm = d_w / torch.sqrt(torch.sum(torch.square(d_w)))
            return d_w_norm

        grad_ws_norm = grad(self.substitute_classifier, torch.as_tensor(x_trigger, device=device, dtype=torch.float32), torch.as_tensor(y_trigger, device=device, dtype=torch.long))

        class NoiseEmbedding(nn.Module):
            def __init__(self, num_poison, len_noise, epsilon, clip_values):
                super(NoiseEmbedding, self).__init__()

                self.embedding_layer = nn.Embedding(num_poison, len_noise)
                self.epsilon = epsilon
                self.clip_values = clip_values

            def forward(self, input_poison, input_indices):
                embeddings = self.embedding_layer(input_indices)
                embeddings = torch.clip(embeddings, -self.epsilon, self.epsilon)
                embeddings = embeddings.view(input_poison.shape)

                input_noised = input_poison + embeddings
                input_noised = torch.clip(input_noised, self.clip_values[0], self.clip_values[1])  # Make sure the poisoned samples are in a valid range.

                return input_noised

        class BackdoorModel(nn.Module):
            def __init__(self, classifier, epsilon, num_poison, len_noise, min_, max_):
                super(BackdoorModel, self).__init__()
                self.classifier = classifier
                self.ne = NoiseEmbedding(num_poison, len_noise, epsilon, (min_, max_))

            def forward(self, x, indices_poison, y, grad_ws_norm):
                # grad_ws_norm = grad(self.classifier, x, y)
                poisoned_samples = self.ne(x, indices_poison)
                d_w2_norm = grad(self.classifier, poisoned_samples, y)
                # grad_ws_norm.requires_grad_(True)
                d_w2_norm.requires_grad_(True)
                B_score = 1 - nn.CosineSimilarity(dim=0)(grad_ws_norm, d_w2_norm)
                return B_score, poisoned_samples

        bm = BackdoorModel(self.classifier, 0.1, num_poison, len_noise, self.clip_values[0], self.clip_values[1])
        optimizer = torch.optim.Adam(bm.ne.embedding_layer.parameters(), lr=1)

        class PoisonDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.len = x.shape[0]
                self.x = torch.as_tensor(x, device=device, dtype=torch.float)
                # self.indices = torch.as_tensor(np.arange(len(y)), device=device, dtype=torch.long)
                self.y = torch.as_tensor(y, device=device, dtype=torch.long)

            def __getitem__(self, index):
                return self.x[index], torch.as_tensor([index], device=device), self.y[index]

            def __len__(self):
                return self.len

        class PredefinedLRSchedule:
            """
            Use a preset learning rate based on the current training epoch.
            """

            def __init__(self, learning_rates, milestones):
                self.schedule = list(zip(milestones, learning_rates))

            def __call__(self, step):
                lr_prev = self.schedule[0][1]
                for m, learning_rate in self.schedule:
                    if step < m:
                        return lr_prev
                    lr_prev = learning_rate
                return lr_prev

            def get_config(self):
                return {"learning_rates": self.learning_rates, "milestones": self.milestones}

        lr_schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            PredefinedLRSchedule(*self.learning_rate_schedule)
            )
        trainloader = torch.utils.data.DataLoader(PoisonDataset(x_poison, y_poison), batch_size=self.batch_size, shuffle=True, num_workers=8)

        epoch_iterator = trange(self.max_epochs) if self.verbose else range(self.max_epochs)
        for _ in epoch_iterator:
            batch_iterator = tqdm(range(trainloader)) if self.verbose else range(trainloader)
            for x, indices, y in batch_iterator:
                bm.zero_grad()
                # loss, poisoned_samples = bm(torch.tensor(x_poison, dtype=torch.float), torch.arange(0, len(x_poison), dtype=torch.int32), torch.tensor(y_poison))
                loss, poisoned_samples = bm(x, indices, y, grad_ws_norm)
                bm.ne.embedding_layer.grad._sign()
                loss.backward()
                optimizer.step()
            lr_schedule.step()

        B, poisoned_samples = bm(torch.tensor(x_poison, dtype=torch.float), torch.arange(0, len(x_poison), dtype=torch.int32), torch.tensor(y_poison))
        raise B.detach().numpy(), poisoned_samples.detach().numpy()

    def __poison__tensorflow(
        self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_poison: np.ndarray, y_poison: np.ndarray
    ) -> np.ndarray:
        """
        Optimize the poison by matching the gradient within the perturbation budget.

        :param x_trigger: List of triggers.
        :param y_trigger: List of target labels.
        :param x_poison: List of samples to poison.
        :param y_poison: List of the labels for x_poison.
        :return: A pair of poisoned samples, B-score (cosine similarity of the gradients).
        """
        import tensorflow.keras.backend as K
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Embedding, Add, Lambda

        # Get the target gradient vector.
        def grad_loss(model, x, target):
            with tf.GradientTape() as t:  # pylint: disable=C0103
                t.watch(model.weights)
                output = model(x)
                loss = model.compiled_loss(target, output)
            d_w = t.gradient(loss, model.trainable_weights)
            d_w = tf.concat([tf.reshape(d, [-1]) for d in d_w], 0)
            d_w_norm = d_w / tf.sqrt(tf.reduce_sum(tf.square(d_w)))
            return d_w_norm

        grad_ws_norm = grad_loss(self.substitute_classifier.model, tf.constant(x_trigger), tf.constant(y_trigger))

        class ClipConstraint(tf.keras.constraints.MaxNorm):
            """
            Clip the tensor values.
            """

            def __init__(self, max_value=2):
                super().__init__(max_value=max_value)

            def __call__(self, w):
                return tf.clip_by_value(w, -self.max_value, self.max_value)

        # Define the model to apply and optimize the poison.
        input_poison = Input(batch_shape=self.substitute_classifier.model.input.shape)
        input_indices = Input(shape=())
        y_true_poison = Input(shape=np.shape(y_poison)[1:])
        embedding_layer = Embedding(len(x_poison), np.prod(input_poison.shape[1:]))
        embeddings = embedding_layer(input_indices)
        embeddings = ClipConstraint(max_value=self.epsilon)(embeddings)
        embeddings = tf.reshape(embeddings, tf.shape(input_poison))
        input_noised = Add()([input_poison, embeddings])
        input_noised = Lambda(lambda x: K.clip(x, self.clip_values[0], self.clip_values[1]))(
            input_noised
        )  # Make sure the poisoned samples are in a valid range.

        def loss_fn(input_noised, target, grad_ws_norm):
            d_w2_norm = grad_loss(self.substitute_classifier.model, input_noised, target)
            B = 1 - tf.reduce_sum(grad_ws_norm * d_w2_norm)  # pylint: disable=C0103
            return B

        B = tf.keras.layers.Lambda(lambda x: loss_fn(x[0], x[1], x[2]))(  # pylint: disable=C0103
            [input_noised, y_true_poison, grad_ws_norm]
        )

        m = tf.keras.models.Model([input_poison, y_true_poison, input_indices], [input_noised, B])
        m.add_loss(B)

        model_trainable = self.substitute_classifier.model.trainable
        self.substitute_classifier.model.trainable = False

        class PredefinedLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            """
            Use a preset learning rate based on the current training epoch.
            """

            def __init__(self, learning_rates, milestones):
                self.schedule = list(zip(milestones, learning_rates))

            def __call__(self, step):
                lr_prev = self.schedule[0][1]
                for m, learning_rate in self.schedule:
                    if step < m:
                        return lr_prev
                    lr_prev = learning_rate
                return lr_prev

            def get_config(self):
                return {"learning_rates": self.learning_rates, "milestones": self.milestones}

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(PredefinedLRSchedule(*self.learning_rate_schedule))

        class SignedAdam(tf.keras.optimizers.Adam):
            """
            This optimizer takes only the sign of the gradients and pass it to the Adam optimizer.
            """

            def compute_gradients(
                self,
                loss,
                var_list=None,
                gate_gradients=1,
                aggregation_method=None,
                colocate_gradients_with_ops=False,
                grad_loss=None,
            ):
                """
                The signs of the gradients are taken and passed to the optimizer.
                """
                grads_and_vars = super().compute_gradients(
                    loss, var_list, gate_gradients, aggregation_method, colocate_gradients_with_ops, grad_loss
                )
                return [(tf.sign(g), v) for (g, v) in grads_and_vars]

        m.compile(loss=None, optimizer=SignedAdam(learning_rate=0.1))

        self.substitute_classifier.model.trainable = model_trainable

        callbacks = [lr_schedule]
        # Train the noise.
        m.fit(
            [x_poison, y_poison, np.arange(len(y_poison))],
            callbacks=callbacks,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            verbose=self.verbose,
        )
        [input_noised_, B_] = m.predict([x_poison, y_poison, np.arange(len(y_poison))])  # pylint: disable=C0103
        return input_noised_, B_

    def _check_params(self) -> None:
        if not isinstance(self.learning_rate_schedule, tuple) or len(self.learning_rate_schedule) != 2:
            raise ValueError("learning_rate_schedule must be a pair of a list of learning rates and a list of epochs")

        if self.percent_poison > 1 or self.percent_poison < 0:
            raise ValueError("percent_poison must be in [0, 1]")

        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")

        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")

        if not isinstance(self.clip_values, tuple) or len(self.clip_values) != 2:
            raise ValueError("clip_values must be a pair (min, max) of floats")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be nonnegative")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if isinstance(self.verbose, int) and self.verbose < 0:
            raise ValueError("verbose must be nonnegative integer or Boolean")
