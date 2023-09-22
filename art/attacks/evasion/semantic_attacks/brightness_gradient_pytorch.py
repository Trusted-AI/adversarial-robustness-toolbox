# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements a semantic attack by adversarially perturbing the brightness component of the inputs. It uses the
iterative gradient sign method to optimise the semantic perturbations (see `FastGradientMethod` and
`BasicIterativeMethod`). This implementation extends the original optimisation method to other norms as well.

| Paper link: https://arxiv.org/abs/2202.04235
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.attacks.evasion.semantic_attacks.hue_gradient_pytorch import HueGradientPyTorch
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class BrightnessGradientPyTorch(HueGradientPyTorch):
    """
    Implementation of the brightness attack on image classifiers in PyTorch. The attack is constructed by adversarially
    perturbing the brightness component of the inputs. It uses the iterative gradient sign method to optimise the
    semantic perturbations (see `FastGradientMethod` and `BasicIterativeMethod`). This implementation extends the
    original optimisation method to other norms as well.

    Note that this attack is intended for only PyTorch image classifiers with RGB images in the range [0, 1] as inputs.

    | Paper link: https://arxiv.org/abs/2202.04235
    """

    attack_params = HueGradientPyTorch.attack_params

    def __init__(
        self,
        classifier: "PyTorchClassifier",
        norm: Union[int, float, str] = np.inf,
        factor_min: float = -0.2,
        factor_max: float = 0.2,
        step_size: Union[int, float] = 0.1,
        max_iter: int = 10,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ) -> None:
        """
        Create an instance of the :class:`.BrightnessGradientPyTorch`.

        :param classifier: A trained PyTorch classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: `"inf"`, `np.inf`, `1` or `2`.
        :param factor_min: The lower bound of the brightness perturbation. The value is expected to be in the interval
                           `[-1, 1]`. Perturbation of `0` means no shift, `-1` gives a complete black image, and `1`
                           gives a complete white image. See `kornia.enhance.adjust_brightness` for more details.
        :param factor_max: The upper bound of the brightness perturbation. The value is expected to be in the interval
                           `[-1, 1]` and greater than `factor_min`.
        :param step_size: Attack step size at each iteration.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (`True`) or untargeted (`False`).
        :param num_random_init: Number of random brightness initialisations to try within the `[factor_min, factor_max]`
                                interval. For `num_random_init=0`, the attack starts at the original input i.e., the
                                initial brightness factor (perturbation) is set to `0`.
        :param batch_size: The batch size to use during the generation of adversarial samples.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in the current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g., pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc., for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        super().__init__(
            classifier=classifier,
            norm=norm,
            factor_min=factor_min,
            factor_max=factor_max,
            step_size=step_size,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            summary_writer=summary_writer,
            verbose=verbose
        )

        self._description = "Brightness Attack"

    def _check_params(self) -> None:

        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')

        if not isinstance(self.factor_min, float) or not -1 <= self.factor_min <= 1:
            raise ValueError("The argument `factor_min` must be in [-1, 1] and of type float.")

        if not isinstance(self.factor_max, float) or not -1 <= self.factor_max <= 1:
            raise ValueError("The argument `factor_max` must be in [-1, 1] and of type float.")

        if self.factor_min >= self.factor_max:
            raise ValueError("The argument `factor_min` must be less than the argument `factor_max`.")

        if not isinstance(self.step_size, (int, float)) or self.step_size <= 0.0:
            raise ValueError("The argument `step_size` must be positive of type int or float.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The argument `max_iter` must be positive of type int.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, int):
            raise TypeError("The argument `num_random_init` has to be of type integer")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size has to be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be a Boolean.")

    def _init_factor(self, x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """
        Initialise the brightness factors.

        :param x: Original inputs.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :return: Initial brightness factors.
        """
        import torch

        shape = torch.Size((x.size(0),))

        if self.num_random_init > 0:
            # Initialise factors to random values sampled from [factor_min, factor_max].
            f_init = torch.distributions.uniform.Uniform(self.factor_min, self.factor_max).sample(shape)
        else:
            # Initialise factors to 0 i.e., no perturbation.
            f_init = torch.zeros(shape)
        f_init = torch.asarray(f_init, dtype=torch.float32, device=self.estimator.device)

        if mask is not None:
            f_init = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), f_init)

        return f_init.clone().detach()

    @staticmethod
    def _apply_factor(x: "torch.Tensor", factor: "torch.Tensor"):
        """
        Compute adversarial samples by applying the current brightness factors.

        :param x: Original inputs.
        :param factor: Current brightness factors.
        :return: A tuple holding the current adversarial examples.
        """
        import kornia

        x_adv = kornia.enhance.adjust_brightness(image=x, factor=factor)

        return x_adv
