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
This module implements STRIP: A Defence Against Trojan Attacks on Deep Neural Networks.

| Paper link: https://arxiv.org/abs/1902.06531
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TypeVar, TYPE_CHECKING

import numpy as np

from art.defences.transformer.transformer import Transformer
from art.estimators.poison_mitigation.strip import STRIPMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

    ClassifierWithStrip = TypeVar("ClassifierWithStrip", CLASSIFIER_TYPE, STRIPMixin)

logger = logging.getLogger(__name__)


class STRIP(Transformer):
    """
    Implementation of STRIP: A Defence Against Trojan Attacks on Deep Neural Networks (Gao et. al. 2020)

    | Paper link: https://arxiv.org/abs/1902.06531
    """

    params = [
        "num_samples",
        "false_acceptance_rate",
    ]

    def __init__(self, classifier: "CLASSIFIER_TYPE"):
        """
        Create an instance of the neural cleanse defence.

        :param classifier: A trained classifier.
        """
        super().__init__(classifier=classifier)
        self._check_params()

    def __call__(  # type: ignore
        self,
        num_samples: int = 20,
        false_acceptance_rate: float = 0.01,
    ) -> "ClassifierWithStrip":
        """
        Create a STRIP defense

        :param num_samples: The number of samples to use to test entropy at inference time
        :param false_acceptance_rate: The percentage of acceptable false acceptance
        """
        base_cls = self.classifier.__class__
        base_cls_name = self.classifier.__class__.__name__
        self.classifier.__class__ = type(
            base_cls_name,
            (STRIPMixin, base_cls),
            dict(
                num_samples=num_samples, false_acceptance_rate=false_acceptance_rate, predict_fn=self.classifier.predict
            ),
        )

        return self.classifier  # type: ignore

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        raise NotImplementedError

    def _check_params(self) -> None:
        pass
