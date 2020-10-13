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

from typing import Union, List

import numpy as np
from matplotlib import pyplot as plt

from art.evaluations.evaluation import Evaluation
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent


class SecurityCurve(Evaluation):
    def __init__(self, eps: Union[int, List[float], List[int]]):

        self.eps = eps
        self.eps_list = list()
        self.accuracy_adv_list = list()
        self.accuracy = None

    def evaluate(self, classifier, x, y, **kwargs):

        kwargs.pop("classifier", None)
        kwargs.pop("eps", None)
        self.eps_list.clear()
        self.accuracy_adv_list.clear()
        self.accuracy = None

        if isinstance(self.eps, int):
            eps_incr = (classifier.clip_values[1] - classifier.clip_values[0]) / self.eps

            for i in range(1, self.eps + 1):
                self.eps_list.append(i * eps_incr)

        else:
            self.eps_list = self.eps.copy()

        y_pred = classifier.predict(x=x, y=y)
        self.accuracy = self._get_accuracy(y=y, y_pred=y_pred)

        for eps in self.eps_list:
            attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=eps, **kwargs)

            x_adv = attack_pgd.generate(x=x, y=y)

            y_pred_adv = classifier.predict(x=x_adv, y=y)
            accuracy_adv = self._get_accuracy(y=y, y_pred=y_pred_adv)
            self.accuracy_adv_list.append(accuracy_adv)

        self._check_gradient(classifier=classifier, x=x, y=y)

        return self.eps_list, self.accuracy_adv_list, self.accuracy

    def _check_gradient(self, classifier, x, y):

        max_iter = 100

        kwargs = {
            "norm": "inf",
            "eps": classifier.clip_values[1],
            "eps_step": classifier.clip_values[1] / (max_iter * 2),
            "max_iter": max_iter,
            "targeted": False,
            "num_random_init": 0,
            "batch_size": 128,
            "random_eps": False,
            "verbose": False,
        }

        attack_pgd = ProjectedGradientDescent(estimator=classifier, **kwargs)

        x_adv = attack_pgd.generate(x=x, y=y)

        y_pred_adv = classifier.predict(x=x_adv, y=y)

        accuracy_adv = self._get_accuracy(y=y, y_pred=y_pred_adv)

        if accuracy_adv > 0.05:
            self.is_obfuscating_gradients = True
        else:
            self.is_obfuscating_gradients = False

    def plot(self):
        plt.plot(self.eps_list, self.accuracy_adv_list, label='adversarial', marker='o')
        plt.plot([self.eps_list[0], self.eps_list[-1]], [self.accuracy, self.accuracy], linestyle="--", label='benign')
        plt.legend()
        plt.xlabel('Attack budget eps')
        plt.ylabel('Accuracy')
        if self.is_obfuscating_gradients:
            plt.title("Potential gradient obfuscation detected.")
        else:
            plt.title("No gradient obfuscation detected")
        plt.ylim([0, 1.05])
        plt.show()

    @staticmethod
    def _get_accuracy(y, y_pred):
        num_data = y.shape[0]
        return np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) / num_data

    def __repr__(self):
        repr_ = "{}(eps={})".format(self.__module__ + "." + self.__class__.__name__, self.eps,)
        return repr_
