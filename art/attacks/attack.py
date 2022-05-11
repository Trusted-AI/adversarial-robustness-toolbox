# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the abstract base classes for all attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import logging
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.exceptions import EstimatorError
from art.summary_writer import SummaryWriter, SummaryWriterDefault

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, GENERATOR_TYPE

logger = logging.getLogger(__name__)


class InputFilter(abc.ABCMeta):  # pragma: no cover
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls
    """

    # pylint: disable=W0613
    def __init__(cls, name, bases, clsdict):  # pylint: disable=W0231
        """
        This function overrides any existing generate or extract methods with a new method that
        ensures the input is an `np.ndarray`. There is an assumption that the input object has implemented
        __array__ with np.array calls.
        """

        def make_replacement(fdict, func_name):
            """
            This function overrides creates replacement functions dynamically
            """

            def replacement_function(self, *args, **kwargs):
                if len(args) > 0:
                    lst = list(args)

                if "x" in kwargs:
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                else:
                    if not isinstance(args[0], np.ndarray):
                        lst[0] = np.array(args[0])

                if "y" in kwargs:
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif len(args) == 2:
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list = ["generate", "extract"]
        for item in replacement_list:
            if item in clsdict:
                new_function = make_replacement(clsdict, item)
                setattr(cls, item, new_function)


class Attack(abc.ABC):
    """
    Abstract base class for all attack abstract base classes.
    """

    attack_params: List[str] = []
    # The _estimator_requirements define the requirements an estimator must satisfy to be used as a target for an
    # attack. They should be a tuple of requirements, where each requirement is either a class the estimator must
    # inherit from, or a tuple of classes which define a union, i.e. the estimator must inherit from at least one class
    # in the requirement tuple.
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None

    def __init__(
        self,
        estimator,
        summary_writer: Union[str, bool, SummaryWriter] = False,
    ):
        """
        :param estimator: An estimator.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        """
        super().__init__()

        if self.estimator_requirements is None:
            raise ValueError("Estimator requirements have not been defined in `_estimator_requirements`.")

        if not self.is_estimator_valid(estimator, self._estimator_requirements):
            raise EstimatorError(self.__class__, self.estimator_requirements, estimator)

        self._estimator = estimator
        self._summary_writer_arg = summary_writer
        self._summary_writer: Optional[SummaryWriter] = None

        if isinstance(summary_writer, SummaryWriter):  # pragma: no cover
            self._summary_writer = summary_writer
        elif summary_writer:
            self._summary_writer = SummaryWriterDefault(summary_writer)

        Attack._check_params(self)

    @property
    def estimator(self):
        """The estimator."""
        return self._estimator

    @property
    def summary_writer(self):
        """The summary writer."""
        return self._summary_writer

    @property
    def estimator_requirements(self):
        """The estimator requirements."""
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: A dictionary of attack-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:

        if not isinstance(self._summary_writer_arg, (bool, str, SummaryWriter)):
            raise ValueError("The argument `summary_writer` has to be either of type bool or str.")

    @staticmethod
    def is_estimator_valid(estimator, estimator_requirements) -> bool:
        """
        Checks if the given estimator satisfies the requirements for this attack.

        :param estimator: The estimator to check.
        :param estimator_requirements: Estimator requirements.
        :return: True if the estimator is valid for the attack.
        """

        for req in estimator_requirements:
            # A requirement is either a class which the estimator must inherit from, or a tuple of classes and the
            # estimator is required to inherit from at least one of the classes
            if isinstance(req, tuple):
                if all(p not in type(estimator).__mro__ for p in req):
                    return False
            elif req not in type(estimator).__mro__:
                return False
        return True


class EvasionAttack(Attack):
    """
    Abstract base class for evasion attack classes.
    """

    def __init__(self, **kwargs) -> None:
        self._targeted = False
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(  # lgtm [py/inheritance/incorrect-overridden-signature]
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        evasion attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
                  or not. This parameter is only used by some of the attacks.
        :return: An array holding the adversarial examples.
        """
        raise NotImplementedError

    @property
    def targeted(self) -> bool:
        """
        Return Boolean if attack is targeted. Return None if not applicable.
        """
        return self._targeted

    @targeted.setter
    def targeted(self, targeted) -> None:
        self._targeted = targeted


class PoisoningAttack(Attack):
    """
    Abstract base class for poisoning attack classes
    """

    def __init__(self, classifier: Optional["CLASSIFIER_TYPE"]) -> None:
        """
        :param classifier: A trained classifier (or none if no classifier is needed)
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the (poisoning examples, poisoning labels).
        """
        raise NotImplementedError


class PoisoningAttackGenerator(Attack):
    """
    Abstract base class for poisoning attack classes that return a transformed generator.
    These attacks have an additional method, `poison_estimator`, that returns the poisoned generator.
    """

    def __init__(self, generator: "GENERATOR_TYPE") -> None:
        """
        :param generator: A generator
        """
        super().__init__(generator)

    @abc.abstractmethod
    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size: int,
        max_iter: int,
        lambda_p: float,
        verbose: int,
        **kwargs
    ) -> "GENERATOR_TYPE":
        """
        Returns a poisoned version of the generator used to initialize the attack
        :return: A poisoned generator
        """
        raise NotImplementedError

    @property
    def z_trigger(self):
        """
        Returns the secret attacker trigger
        """
        return self._z_trigger

    @property
    def x_target(self):
        """
        Returns the secret attacker target which the poisoned generator should produce
        """
        return self._x_target


class PoisoningAttackTransformer(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that return a transformed classifier.
    These attacks have an additional method, `poison_estimator`, that returns the poisoned classifier.
    """

    def __init__(self, classifier: Optional["CLASSIFIER_TYPE"]) -> None:
        """
        :param classifier: A trained classifier (or none if no classifier is needed)
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the (poisoning examples, poisoning labels).
        :rtype: `(np.ndarray, np.ndarray)`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def poison_estimator(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "CLASSIFIER_TYPE":
        """
        Returns a poisoned version of the classifier used to initialize the attack
        :param x: Training data
        :param y: Training labels
        :return: A poisoned classifier
        """
        raise NotImplementedError


class PoisoningAttackBlackBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have no access to the model (classifier object).
    """

    def __init__(self):
        """
        Initializes black-box data poisoning attack.
        """
        super().__init__(None)  # type: ignore

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        raise NotImplementedError


class PoisoningAttackWhiteBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have white-box access to the model (classifier object).
    """

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        raise NotImplementedError


class ExtractionAttack(Attack):
    """
    Abstract base class for extraction attack classes.
    """

    @abc.abstractmethod
    def extract(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> "CLASSIFIER_TYPE":
        """
        Extract models and return them as an ART classifier. This method should be overridden by all concrete extraction
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :return: ART classifier of the extracted model.
        """
        raise NotImplementedError


class InferenceAttack(Attack):
    """
    Abstract base class for inference attack classes.
    """

    def __init__(self, estimator):
        """
        :param estimator: A trained estimator targeted for inference attack.
        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`
        """
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer sensitive attributes from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param x: An array with reference inputs to be used in the attack.
        :param y: Labels for `x`. This parameter is only used by some of the attacks.
        :return: An array holding the inferred attribute values.
        """
        raise NotImplementedError


class AttributeInferenceAttack(InferenceAttack):
    """
    Abstract base class for attribute inference attack classes.
    """

    attack_params = InferenceAttack.attack_params + ["attack_feature"]

    def __init__(self, estimator, attack_feature: Union[int, slice] = 0):
        """
        :param estimator: A trained estimator targeted for inference attack.
        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`
        :param attack_feature: The index of the feature to be attacked.
        """
        super().__init__(estimator)
        self.attack_feature = attack_feature

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer sensitive attributes from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param x: An array with reference inputs to be used in the attack.
        :param y: Labels for `x`. This parameter is only used by some of the attacks.
        :return: An array holding the inferred attribute values.
        """
        raise NotImplementedError


class MembershipInferenceAttack(InferenceAttack):
    """
    Abstract base class for membership inference attack classes.
    """

    def __init__(self, estimator):
        """
        :param estimator: A trained estimator targeted for inference attack.
        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`
        :param attack_feature: The index of the feature to be attacked.
        """
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership status of samples from the target estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param x: An array with reference inputs to be used in the attack.
        :param y: Labels for `x`. This parameter is only used by some of the attacks.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class.
        :return: An array holding the inferred membership status (1 indicates member of training set,
                 0 indicates non-member) or class probabilities.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.
        """
        # Save attack-specific parameters
        super().set_params(**kwargs)
        self._check_params()


class ReconstructionAttack(Attack):
    """
    Abstract base class for reconstruction attack classes.
    """

    attack_params = InferenceAttack.attack_params

    def __init__(self, estimator):
        """
        :param estimator: A trained estimator targeted for reconstruction attack.
        """
        super().__init__(estimator)

    @abc.abstractmethod
    def reconstruct(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the training dataset of and from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param x: An array with known records of the training set of `estimator`.
        :param y: An array with known labels of the training set of `estimator`, if None predicted labels will be used.
        :return: A tuple of two arrays for the reconstructed training input and labels.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.
        """
        # Save attack-specific parameters
        super().set_params(**kwargs)
        self._check_params()
