# todo: license

"""
todo: summary 
"""

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

class SignOPTAttack(EvasionAttack):
    """[summary]

    Args:
        EvasionAttack ([type]): [description]
    | todo: paper link: 
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted", 
        "verbose", 
        # todo: add others
    ]

    def __init__(
        self, 
        estimator: "CLASSIFIER_TYPE",
        verbose: bool = True,
    ) -> None:
        """
        Create a Sign_OPT attack instance.

        :param estimator: A trained classifier.
        :param targeted: Should the attack target one specific class.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)
        self._targeted = targeted

        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return super().generate(x, y=y, **kwargs)

    def _check_params(self) -> None:
        # todo: add param checking
        return super()._check_params()