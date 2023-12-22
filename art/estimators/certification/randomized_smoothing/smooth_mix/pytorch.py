# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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

# MIT License
#
# Copyright (c) 2021 Jongheon Jeong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements SmoothMix applied to classifier predictions.

| Paper link: https://arxiv.org/abs/2111.09277
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from tqdm.auto import trange
import numpy as np

from art.estimators.certification.randomized_smoothing.pytorch import PyTorchRandomizedSmoothing
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchSmoothMix(PyTorchRandomizedSmoothing):
    """
    Implementation of SmoothMix training, as introduced in Jeong et al. (2021)

    | Paper link: https://arxiv.org/abs/2111.09277
    """

    estimator_params = PyTorchRandomizedSmoothing.estimator_params + [
        "eta",
        "num_noise_vec",
        "num_steps",
        "warmup",
        "mix_step",
        "maxnorm_s",
        "maxnorm",
    ]

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
        sample_size: int = 32,
        scale: float = 0.1,
        alpha: float = 0.001,
        eta: float = 1.0,
        num_noise_vec: int = 1,
        num_steps: int = 10,
        warmup: int = 1,
        mix_step: int = 0,
        maxnorm_s: Optional[float] = None,
        maxnorm: Optional[float] = None,
    ) -> None:
        """
        Create a SmoothMix classifier.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param optimizer: The optimizer used to train the classifier.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param sample_size: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
        :param eta: The relative strength of the mixup loss.
        :param num_noise_vec: The number of noise vectors.
        :param num_steps: The number of attack updates.
        :param warmup: The warm-up strategy that is gradually increased up to the original value.
        :param mix_step: Determines which sample to use for the clean side.
        :param maxnorm_s: The initial value of `alpha * mix_step`.
        :param maxnorm: The initial value of `alpha * mix_step` for adversarial examples.
        """
        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
        )
        self.eta = eta
        self.num_noise_vec = num_noise_vec
        self.num_steps = num_steps
        self.warmup = warmup
        self.mix_step = mix_step
        self.maxnorm_s = maxnorm_s
        self.maxnorm = maxnorm

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        drop_last: bool = False,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param verbose: Display the training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        # Set model mode
        self._model.train(mode=training_mode)

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        # Create dataloader
        x_tensor = torch.from_numpy(x_preprocessed)
        y_tensor = torch.from_numpy(y_preprocessed)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        # Start training
        for epoch in trange(nb_epochs, disable=not verbose):
            warmup_v = min(1.0, (epoch + 1) / self.warmup)

            for x_batch, y_batch in dataloader:
                mini_batch_size = len(x_batch) // self.num_noise_vec

                for mini_batch in range(self.num_noise_vec):
                    # Create mini batch
                    inputs = x_batch[mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size]
                    labels = y_batch[mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size]

                    # Move inputs to GPU
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    noises = [torch.randn_like(inputs) * self.scale for _ in range(self.num_noise_vec)]

                    # Attack and find adversarial examples
                    self._model.eval()
                    inputs, inputs_adv = self._smooth_mix_pgd_attack(inputs, labels, noises, warmup_v)
                    self._model.train(mode=training_mode)

                    in_clean_c = torch.cat([inputs + noise for noise in noises], dim=0)
                    logits_c = self._model(in_clean_c)[-1]
                    labels_c = labels.repeat(self.num_noise_vec)

                    logits_c_chunk = torch.chunk(logits_c, self.num_noise_vec, dim=0)
                    clean_sm = F.softmax(torch.stack(logits_c_chunk), dim=-1)
                    clean_avg_sm = torch.mean(clean_sm, dim=0)
                    loss_xent = F.cross_entropy(logits_c, labels_c, reduction="none")

                    # mix adversarial examples
                    in_mix, labels_mix = self._mix_data(inputs, inputs_adv, clean_avg_sm)
                    in_mix_c = torch.cat([in_mix + noise for noise in noises], dim=0)
                    labels_mix_c = labels_mix.repeat(self.num_noise_vec, 1)
                    logits_mix_c = F.log_softmax(self._model(in_mix_c)[-1], dim=1)

                    preds = torch.argmax(clean_avg_sm, dim=-1)
                    ind_correct = (preds == labels).float()
                    ind_correct = ind_correct.repeat(self.num_noise_vec)

                    loss_mixup = F.kl_div(logits_mix_c, labels_mix_c, reduction="none").sum(1)
                    loss = loss_xent.mean() + self.eta * warmup_v * (ind_correct * loss_mixup).mean()

                    # compute gradient and do SGD step
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

            if scheduler is not None:
                scheduler.step()

    def _smooth_mix_pgd_attack(
        self,
        inputs: "torch.Tensor",
        labels: "torch.Tensor",
        noises: List["torch.Tensor"],
        warmup_v: float,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        The authors' implementation of the SmoothMixPGD attack.
        Code modified from https://github.com/jh-jeong/smoothmix/blob/main/code/train.py

        :param inputs: The batch inputs
        :param labels: The batch labels for the inputs
        :param noises: The noise applied to each input in the attack
        """
        import torch
        import torch.nn.functional as F

        def _batch_l2_norm(x: torch.Tensor) -> torch.Tensor:
            """
            Perform a batch L2 norm

            :param x: The inputs to compute the batch L2 norm of
            """
            x_flat = x.reshape(x.size(0), -1)
            return torch.norm(x_flat, dim=1)

        def _project(x: torch.Tensor, x_0: torch.Tensor, maxnorm: Optional[float] = None):
            """
            Apply a projection of the current inputs with the maxnorm

            :param x: The inputs to apply a projection on (either original or adversarial)
            :param x_0: The unperterbed inputs to apply the projection on
            :param maxnorm: The maxnorm value to apply to x
            """
            if maxnorm is not None:
                eta = x - x_0
                eta = eta.renorm(p=2, dim=0, maxnorm=maxnorm)
                x = x_0 + eta
            x = torch.clamp(x, 0, 1)
            x = x.detach()
            return x

        adv = inputs.detach()
        init = inputs.detach()
        for i in range(self.num_steps):
            if i == self.mix_step:
                init = adv.detach()
            adv.requires_grad_()

            softmax = [F.softmax(self._model(adv + noise)[-1], dim=1) for noise in noises]
            avg_softmax = torch.mean(torch.stack(softmax), dim=0)
            log_softmax = torch.log(avg_softmax.clamp(min=1e-20))
            loss = F.nll_loss(log_softmax, labels, reduction="sum")

            grad = torch.autograd.grad(loss, [adv])[0]
            grad_norm = _batch_l2_norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)
            adv = adv + self.alpha * grad

            adv = _project(adv, inputs, self.maxnorm)

        if self.maxnorm_s is None:
            maxnorm_s = self.alpha * self.mix_step * warmup_v
        else:
            maxnorm_s = self.maxnorm_s * warmup_v
        init = _project(init, inputs, maxnorm_s)

        return init, adv

    def _mix_data(
        self, inputs: "torch.Tensor", inputs_adv: "torch.Tensor", labels: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Returns mixed inputs and labels.

        :param inputs: Training data
        :param inputs_adv: Adversarial training data
        :param labels: Training labels
        """
        import torch

        eye = torch.eye(self.nb_classes, device=self.device)
        unif = eye.mean(0, keepdim=True)
        lam = torch.rand(inputs.size(0), device=self.device) / 2

        mixed_inputs = (1 - lam).view(-1, 1, 1, 1) * inputs + lam.view(-1, 1, 1, 1) * inputs_adv
        mixed_labels = (1 - lam).view(-1, 1) * labels + lam.view(-1, 1) * unif

        return mixed_inputs, mixed_labels
