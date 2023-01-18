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
This module implements the Patch-Fool attack in PyTorch.

| Paper link: https://arxiv.org/abs/2203.08392
"""
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from art.attacks.attack import EvasionAttack
from art.estimators.classification.pytorch import PyTorchClassifierViT
from art.utils import get_labels_np_array


class PatchFool(EvasionAttack):
    """
    This class represents a Patch-Fool evasion attack in PyTorch.

    | Paper link: https://arxiv.org/abs/2203.08392
    """

    attack_params = EvasionAttack.attack_params

    _estimator_requirements = (PyTorchClassifierViT,)

    def __init__(
        self,
        estimator: "PYTORCH_ESTIMATOR_TYPE",
        alpha: float = 0.002,
        max_iter: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.2,
        step_size: Union[int, float] = 10,
        step_size_decay: Union[int, float] = 0.95,
        patch_layer: int = 4,
        random_start: bool = False,
    ):
        """
        Create a :class:`PatchFool` instance.
        TODO
        """
        if not estimator.all_framework_preprocessing:
            raise NotImplementedError(
                "The framework-specific implementation only supports framework-specific preprocessing."
            )

        super().__init__(estimator=estimator)

        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.step_size_decay = step_size_decay
        self.patch_layer = patch_layer
        self.random_start = random_start
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: Source samples.
        :param y: Target labels.
        :return: Adversarial examples.
        """
        import torch

        nb_samples = x.shape[0]
        x_adv = [None] * nb_samples
        nb_batches = int(np.ceil(nb_samples / float(self.batch_size)))

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        for idx in range(nb_batches):

            begin, end = idx * self.batch_size, min((idx + 1) * self.batch_size, nb_samples)

            x_batch = torch.tensor(x[begin:end], dtype=torch.float32)
            y_batch = torch.tensor(y[begin:end], dtype=torch.float32)

            x_adv[begin:end] = self._generate_batch(x_batch, y_batch).cpu().detach().numpy()

        return np.array(x_adv, dtype=np.float32)

    def _generate_batch(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        TODO
        """
        import torch

        x = x.to(self.estimator.device)
        y = y.to(self.estimator.device)

        p = self._get_patch_index(x, layer=self.patch_layer)

        patch_size = self.estimator.patch_size
        mask = torch.zeros(x.shape).to(self.estimator.device)

        for n, patch_idx in enumerate(p):
            row = (patch_idx // (x.shape[2] // patch_size)) * patch_size
            col = (patch_idx % (x.shape[2] // patch_size)) * patch_size
            mask[n, :, row : row + patch_size, col : col + patch_size] = 1

        x_adv = torch.clone(x).to(self.estimator.device)
        x_adv = torch.mul(x_adv, 1 - mask)
        if self.random_start:
            perturbation = torch.rand(x.shape).to(self.estimator.device)
        else:
            perturbation = torch.zeros(x.shape).to(self.estimator.device)
        perturbation.requires_grad = True

        optim = torch.optim.Adam([perturbation], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=self.step_size, gamma=self.step_size_decay)

        x_adv += torch.mul(perturbation, mask)

        for i_max_iter in range(self.max_iter):
            optim.zero_grad()

            loss_att = self._get_attention_loss(x_adv, p)
            loss_att_batch = torch.sum(loss_att, dim=1)

            model_outputs, _ = self.estimator._predict_framework(x_adv)
            loss_ce = self.estimator._loss(model_outputs, y)

            grad_ce = torch.autograd.grad(loss_ce, perturbation, retain_graph=True)[0]
            grad_att = torch.autograd.grad(torch.sum(loss_att_batch), perturbation, retain_graph=True)[0]

            # Reshape for PCgrad
            grad_att_tmp = grad_att.reshape(grad_att.shape[0], -1)
            grad_ce_tmp = grad_ce.reshape(grad_ce.shape[0], -1)
            grad_att_tmp = self.pcgrad(grad_att_tmp, grad_ce_tmp)
            grad_att = grad_att_tmp.reshape(grad_att.shape)

            grad_ce += self.alpha * grad_att
            perturbation.grad = -grad_ce
            optim.step()
            scheduler.step()

            x_adv += torch.mul(perturbation, mask)

        return x_adv

    def _get_patch_index(self, x: "torch.Tensor", layer: int) -> "torch.Tensor":
        """
        Select the most influencial patch according to a predefined `layer`.
        TODO
        """
        import torch

        att = self.estimator.get_attention_weights(x)
        s_l = torch.sum(att[:, layer, ...], dim=1)
        patch_idx = torch.argmax(s_l, dim=1)

        return patch_idx

    def _get_attention_loss(self, x: "torch.Tensor", patch_idx: "torch.Tensor") -> "torch.Tensor":
        """
        Sum the attention weights from each layer for the most influencail patches
        TODO
        """
        import torch

        att = self.estimator.get_attention_weights(x)
        att_loss = [torch.sum(att[i, :, :, idx], dim=1) for i, idx in enumerate(patch_idx)]

        return torch.stack(att_loss)

    def pcgrad(self, grad1, grad2):
        """
        TODO
        """
        import torch

        cos_sim = torch.nn.functional.cosine_similarity(grad1, grad2)

        grad1_tmp = grad1[cos_sim < 0]
        grad2_tmp = grad2[cos_sim < 0]
        dot_prod = torch.mul(grad1_tmp, grad2_tmp).sum(dim=1)
        beta = (dot_prod / torch.norm(grad2_tmp, dim=1)).view(-1, 1)
        pcgrad = grad1_tmp - beta * grad2_tmp

        grad1[cos_sim < 0] = pcgrad

        return grad1

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if not isinstance(self.alpha, (int, float)):
            raise TypeError("The weight coefficient `alpha` must be of type `int` or `float`.")

        if self.max_iter < 0:
            raise ValueError("The number of iterations `max_iter` has to be a non-negative integer.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.step_size, int):
            raise TypeError("The step size must be of type `int`.")

        if not isinstance(self.step_size_decay, (int, float)):
            raise TypeError("The step size decay coefficient must be of type `int` or `float`.")

        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError("The learning rate must be of type `int` or `float`.")
