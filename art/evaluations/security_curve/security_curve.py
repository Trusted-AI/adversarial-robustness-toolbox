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
This module implements the evaluation of Security Curves.

Examples of Security Curves can be found in Figure 6 of Madry et al., 2017 (https://arxiv.org/abs/1706.06083).
"""
from typing import List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from art.evaluations.evaluation import Evaluation
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


class SecurityCurve(Evaluation):
    """
    This class implements the evaluation of Security Curves.

    Examples of Security Curves can be found in Figure 6 of Madry et al., 2017 (https://arxiv.org/abs/1706.06083).
    """

    def __init__(self, eps: Union[int, List[float], List[int]]):
        """
        Create an instance of a Security Curve evaluation.

        :param eps: Defines the attack budgets `eps` for Projected Gradient Descent used for evaluation.
        """

        self.eps = eps
        self.eps_list: List[float] = list()
        self.accuracy_adv_list: List[float] = list()
        self.accuracy: Optional[float] = None

    # pylint: disable=W0221
    def evaluate(  # type: ignore
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Union[str, bool, int, float]
    ) -> Tuple[List[float], List[float], float]:
        """
        Evaluate the Security Curve of a classifier using Projected Gradient Descent.

        :param classifier: A trained classifier that provides loss gradients.
        :param x: Input data to classifier for evaluation.
        :param y: True labels for input data `x`.
        :param kwargs: Keyword arguments for the Projected Gradient Descent attack used for evaluation, except keywords
                       `classifier` and `eps`.
        :return: List of evaluated `eps` values, List of adversarial accuracies, and benign accuracy.
        """

        kwargs.pop("classifier", None)
        kwargs.pop("eps", None)
        self.eps_list.clear()
        self.accuracy_adv_list.clear()
        self.accuracy = None

        # Check type of eps
        if isinstance(self.eps, int):
            if classifier.clip_values is not None:
                eps_increment = (classifier.clip_values[1] - classifier.clip_values[0]) / self.eps
            else:
                eps_increment = (np.max(x) - np.min(x)) / self.eps

            for i in range(1, self.eps + 1):
                self.eps_list.append(float(i * eps_increment))

        else:
            self.eps_list = [float(eps) for eps in self.eps]

        # Determine benign accuracy
        y_pred = classifier.predict(x=x, y=y)
        self.accuracy = self._get_accuracy(y=y, y_pred=y_pred)

        # Determine adversarial accuracy for each eps
        for eps in self.eps_list:
            attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=eps, **kwargs)  # type: ignore

            x_adv = attack_pgd.generate(x=x, y=y)

            y_pred_adv = classifier.predict(x=x_adv, y=y)
            accuracy_adv = self._get_accuracy(y=y, y_pred=y_pred_adv)
            self.accuracy_adv_list.append(accuracy_adv)

        # Check gradients for potential obfuscation
        self._check_gradient(classifier=classifier, x=x, y=y, **kwargs)

        return self.eps_list, self.accuracy_adv_list, self.accuracy

    @property
    def detected_obfuscating_gradients(self) -> bool:
        """
        This property describes if the previous call to method `evaluate` identified potential gradient obfuscation.
        """
        return self._detected_obfuscating_gradients

    def _check_gradient(
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Union[str, bool, int, float]
    ) -> None:
        """
        Check if potential gradient obfuscation can be detected. Projected Gradient Descent with 100 iterations is run
        with maximum attack budget `eps` being equal to upper clip value of input data and `eps_step` of
        `eps / (max_iter / 2)`.

        :param classifier: A trained classifier that provides loss gradients.
        :param x: Input data to classifier for evaluation.
        :param y: True labels for input data `x`.
        :param kwargs: Keyword arguments for the Projected Gradient Descent attack used for evaluation, except keywords
                       `classifier` and `eps`.
        """
        # Define parameters for Projected Gradient Descent
        max_iter = 100
        kwargs["max_iter"] = max_iter
        if classifier.clip_values is not None:
            clip_value_max = classifier.clip_values[1]
        else:
            clip_value_max = np.max(x)
        kwargs["eps"] = float(clip_value_max)
        kwargs["eps_step"] = float(clip_value_max / (max_iter / 2))

        # Create attack
        attack_pgd = ProjectedGradientDescent(estimator=classifier, **kwargs)  # type: ignore

        # Evaluate accuracy with maximal attack budget
        x_adv = attack_pgd.generate(x=x, y=y)
        y_pred_adv = classifier.predict(x=x_adv, y=y)
        accuracy_adv = self._get_accuracy(y=y, y_pred=y_pred_adv)

        # Decide of obfuscated gradients likely
        if accuracy_adv > 1 / classifier.nb_classes:
            self._detected_obfuscating_gradients = True
        else:
            self._detected_obfuscating_gradients = False

    def plot(self) -> None:
        """
        Plot the Security Curve of adversarial accuracy as function opf attack budget `eps` together with the accuracy
        on benign samples.
        """
        from matplotlib import pyplot as plt

        plt.plot(self.eps_list, self.accuracy_adv_list, label="adversarial", marker="o")
        plt.plot([self.eps_list[0], self.eps_list[-1]], [self.accuracy, self.accuracy], linestyle="--", label="benign")
        plt.legend()
        plt.xlabel("Attack budget eps")
        plt.ylabel("Accuracy")
        if self.detected_obfuscating_gradients:
            plt.title("Potential gradient obfuscation detected.")
        else:
            plt.title("No gradient obfuscation detected")
        plt.ylim([0, 1.05])
        plt.show()

    @staticmethod
    def _get_accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy of predicted labels.

        :param y: True labels.
        :param y_pred: Predicted labels.
        :return: Accuracy.
        """
        return np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)).item()

    def __repr__(self):
        repr_ = "{}(eps={})".format(
            self.__module__ + "." + self.__class__.__name__,
            self.eps,
        )
        return repr_
