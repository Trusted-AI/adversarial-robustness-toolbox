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
This module implements the universal adversarial perturbations attack `TargetedUniversalPerturbation`.

| Paper link: https://arxiv.org/abs/1911.06502
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import types
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

import numpy as np

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import projection

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class TargetedUniversalPerturbation(EvasionAttack):
    """
    Implementation of the attack from Hirano and Takemoto (2019). Computes a fixed perturbation to be applied to all
    future inputs. To this end, it can use any adversarial attack method.

    | Paper link: https://arxiv.org/abs/1911.06502
    """

    attacks_dict = {
        "fgsm": "art.attacks.evasion.fast_gradient.FastGradientMethod",
        "simba": "art.attacks.evasion.simba.SimBA",
    }
    attack_params = EvasionAttack.attack_params + ["attacker", "attacker_params", "delta", "max_iter", "eps", "norm"]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        attacker: str = "fgsm",
        attacker_params: Optional[Dict[str, Any]] = None,
        delta: float = 0.2,
        max_iter: int = 20,
        eps: float = 10.0,
        norm: Union[int, float, str] = np.inf,
    ):
        """
        :param classifier: A trained classifier.
        :param attacker: Adversarial attack name. Default is 'deepfool'. Supported names: 'fgsm'.
        :param attacker_params: Parameters specific to the adversarial attack. If this parameter is not specified,
                                the default parameters of the chosen attack will be used.
        :param delta: desired accuracy
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :param eps: Attack step size (input variation)
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 2
        """
        super().__init__(estimator=classifier)

        self.attacker = attacker
        self.attacker_params = attacker_params
        self.delta = delta
        self.max_iter = max_iter
        self.eps = eps
        self.norm = norm
        self._targeted = True
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: An array with the targeted labels.
        :return: An array holding the adversarial examples.
        """
        if y is None:
            raise ValueError("Labels `y` cannot be None.")

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        logger.info("Computing targeted universal perturbation based on %s attack.", self.attacker)

        # Init universal perturbation
        noise = 0
        fooling_rate = 0.0
        targeted_success_rate = 0.0
        nb_instances = len(x)

        # Instantiate the middle attacker and get the predicted labels
        attacker = self._get_attack(self.attacker, self.attacker_params)
        pred_y = self.estimator.predict(x, batch_size=1)
        pred_y_max = np.argmax(pred_y, axis=1)

        # Start to generate the adversarial examples
        nb_iter = 0
        while targeted_success_rate < 1.0 - self.delta and nb_iter < self.max_iter:
            # Go through all the examples randomly
            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for _, (e_x, e_y) in enumerate(zip(x[rnd_idx], y[rnd_idx])):
                x_i = e_x[None, ...]
                y_i = e_y[None, ...]

                current_label = np.argmax(self.estimator.predict(x_i + noise)[0])
                target_label = np.argmax(y_i)

                if current_label != target_label:
                    # Compute adversarial perturbation
                    adv_xi = attacker.generate(x_i + noise, y=y_i)

                    new_label = np.argmax(self.estimator.predict(adv_xi)[0])

                    # If the class has changed, update v
                    if new_label == target_label:
                        noise = adv_xi - x_i

                        # Project on L_p ball
                        noise = projection(noise, self.eps, self.norm)
            nb_iter += 1

            # Apply attack and clip
            x_adv = x + noise
            if hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)

            # Compute the error rate
            y_adv = np.argmax(self.estimator.predict(x_adv, batch_size=1), axis=1)
            fooling_rate = np.sum(pred_y_max != y_adv) / nb_instances
            targeted_success_rate = np.sum(y_adv == np.argmax(y, axis=1)) / nb_instances

        self.fooling_rate = fooling_rate
        self.targeted_success_rate = targeted_success_rate
        self.converged = nb_iter < self.max_iter
        self.noise = noise
        logger.info("Fooling rate of universal perturbation attack: %.2f%%", 100 * fooling_rate)
        logger.info("Targeted success rate of universal perturbation attack: %.2f%%", 100 * targeted_success_rate)

        return x_adv

    def _check_params(self) -> None:

        if not isinstance(self.delta, (float, int)) or self.delta < 0 or self.delta > 1:
            raise ValueError("The desired accuracy must be in the range [0, 1].")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.eps, (float, int)) or self.eps <= 0:
            raise ValueError("The eps coefficient must be a positive float.")

    def _get_attack(self, a_name: str, params: Optional[Dict[str, Any]] = None) -> EvasionAttack:
        """
        Get an attack object from its name.

        :param a_name: attack name.
        :param params: attack params.
        :return: attack object
        """
        try:
            attack_class = self._get_class(self.attacks_dict[a_name])
            a_instance = attack_class(self.estimator)  # type: ignore

            if params:
                a_instance.set_params(**params)

            return a_instance

        except KeyError:
            raise NotImplementedError("{} attack not supported".format(a_name)) from KeyError

    @staticmethod
    def _get_class(class_name: str) -> types.ModuleType:
        """
        Get a class module from its name.

        :param class_name: Full name of a class.
        :return: The class `module`.
        """
        sub_mods = class_name.split(".")
        module_ = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
        class_module = getattr(module_, sub_mods[-1])

        return class_module
