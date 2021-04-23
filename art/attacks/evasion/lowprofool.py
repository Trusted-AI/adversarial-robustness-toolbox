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
This module implements the `LowProFool` attack. This is a white-box attack.

Its main objective is to take a valid tabular sample and transform it, so that a given classifier predicts it to be some
target class.

`LowProFool` attack transforms the provided real-valued tabular data into adversaries of the specified target classes.
The generated adversaries have to be as close as possible to the original samples in terms of the weighted Lp-norm,
where the weights determine each feature's importance.

| Paper link: https://arxiv.org/abs/1911.03274
"""
import logging
import numpy as np
from scipy.stats import pearsonr
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import LossGradientsMixin
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class LowProFool(EvasionAttack):
    """
    `LowProFool` attack.

    | Paper link: https://arxiv.org/abs/1911.03274
    """
    attack_params = EvasionAttack.attack_params + [
        "n_steps",
        "threshold",
        "lambd",
        "eta",
        "eta_decay",
        "eta_min",
        "p",
        "importance",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
            self,
            classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
            n_steps: Optional[Optional[int]] = 100,
            threshold: Optional[Optional[Union[float, None]]] = 0.5,
            lambd: Optional[float] = 1.5,
            eta: Optional[float] = 0.2,
            eta_decay: Optional[float] = 0.98,
            eta_min: Optional[float] = 1e-7,
            p: Optional[Union[int, float, str]] = 2,
            importance: Optional[Union[Callable, str, np.ndarray]] = 'pearson',
            verbose: Optional[bool] = False
    ) -> None:
        """
        Create a LowProFool instance.

        :param classifier: Appropriate classifier's instance
        :param n_steps: Number of iterations to follow
        :param threshold: Lowest prediction probability of a valid adversary
        :param lambd: Amount of lp-norm impact on objective function
        :param eta: Rate of updating the perturbation vectors
        :param eta_decay: Step-by-step decrease of eta
        :param eta_min: Minimal eta value
        :param p: Parameter `p` for Lp-space norm (p=2 - euclidean norm)
        :param importance: Function to calculate feature importance with
            or vector of those precomputed; possibilities:
            > 'pearson' - Pearson correlation (string)
            > function  - Custom function (callable object)
            > vector    - Vector of feature importance (np.ndarray)
        :param verbose: Verbose mode / Show progress bars.
        """
        super().__init__(estimator=classifier)

        self.n_steps = n_steps
        self.threshold = threshold
        self.lambd = lambd
        self.eta = eta
        self.eta_decay = eta_decay
        self.eta_min = eta_min
        self.p = p
        self.importance = importance
        self.verbose = verbose

        self._targeted = True
        self.n_classes = self.estimator.nb_classes
        self.n_features = self.estimator.input_shape[0]
        self.importance_vec = None
        if self.estimator.clip_values is None:
            logger.warning(
                "The `clip_values` attribute of the estimator is `None`, therefore this instance of LowProFool will by "
                "default generate adversarial perturbations without clipping them."
            )

        self._check_params()

        if isinstance(self.importance, np.ndarray):
            self.importance_vec = self.importance

        if eta_decay < 1 and eta_min > 0:
            steps_before_min_eta_reached = np.ceil(np.log(eta_min / eta) / np.log(eta_decay))

            if steps_before_min_eta_reached / self.n_steps < 0.8:
                logger.warning(
                    "The given combination of 'n_steps', 'eta', 'eta_decay' and 'eta_min' effectively sets learning "
                    "rate to its minimal value after about {} steps out of all {}.".format(
                        steps_before_min_eta_reached, self.n_steps
                    )
                )

    @staticmethod
    def logistic_loss(
            true: np.ndarray,
            predicted: np.ndarray
    ) -> np.ndarray:
        """
        Logistic loss function. Produces average cross-entropy in the sample.

        :param true: Ground truth (correct) labels for samples.
        :param predicted: Predicted probabilities.
        :return: Mean cross entropy.
        """
        p = np.array([1 - true, true])
        q = np.array([1 - predicted, predicted])

        return np.mean(
            -1 * np.sum(p * np.log(q + 1e-13), axis=0), axis=1
        ).reshape(-1, 1)

    def __weighted_lp_norm(
            self,
            perturbations: np.ndarray
    ) -> np.ndarray:
        """
        Lp-norm of perturbation vectors weighted by feature importance.

        :param perturbations: Perturbations of samples towards being adversarial.
        :return: Array with weighted Lp-norm of perturbations.
        """
        return self.lambd * np.linalg.norm(
            self.importance_vec * perturbations, axis=1,
            ord=(np.inf if self.p == "inf" else self.p)
        ).reshape(-1, 1)

    def __classification_loss_gradient(
            self,
            samples: np.ndarray,
            perturbations: np.ndarray,
            y_probas: np.ndarray,
            targets: np.ndarray
    ) -> np.ndarray:
        """
        Obtain the loss gradients of the underlying classifier with regards to the data vectors.

        :param samples: Base design matrix.
        :param perturbations: Perturbations of samples towards being adversarial.
        :param y_probas: Class-wise prediction probabilities.
        :param targets: The target labels for the attack.
        :return: Array of classification loss gradients.
        """
        return self.estimator.loss_gradient(
            (samples + perturbations).astype(np.float32),
            (targets).astype(np.float32)
        )

    def __weighted_lp_norm_gradient(
            self,
            perturbations: np.ndarray
    ) -> np.ndarray:
        """
        Gradient of the weighted Lp-space norm with regards to the data vector.

        :param perturbations: Perturbations of samples towards being adversarial.
        :return: Weighted Lp-norm gradients array.
        """
        p = self.p
        v = self.importance_vec
        x = perturbations

        if p in ["inf", np.inf]:
            numerator = np.array(v * x)
            optimum = np.max(np.abs(numerator))
            return np.where(abs(numerator) == optimum, np.sign(numerator), 0)

        else:
            numerator = v * v * x * np.power(np.abs(x), p - 2)
            denominator = np.power(np.sum(np.power(v * x, p)), (p - 1) / p)
            return (
                np.where(denominator > 1e-10, numerator, np.zeros(numerator.shape[1])) /
                np.where(denominator <= 1e-10, 1., denominator)
            )

    def __get_gradients(
            self,
            samples: np.ndarray,
            perturbations: np.ndarray,
            y_probas: np.ndarray,
            targets: np.ndarray
    ) -> np.ndarray:
        """
        Gradient of the objective function with regards to the data vector, i.e. sum of the classifier's loss gradient
        and weighted lp-space norm gradient, both with regards to data vector.

        :param samples: Base design matrix.
        :param perturbations: Perturbations of samples towards being adversarial.
        :param y_probas: Class-wise prediction probabilities.
        :param targets: The target labels for the attack.
        :return: Aggregate gradient of objective function.
        """
        clf_loss_grad = self.__classification_loss_gradient(samples, perturbations, y_probas, targets)
        norm_grad = self.lambd * self.__weighted_lp_norm_gradient(perturbations)

        return clf_loss_grad + norm_grad

    def __loss_function(
            self,
            y_probas: np.ndarray,
            perturbations: np.ndarray,
            targets: np.ndarray
    ) -> np.ndarray:
        """
        Complete loss function to optimize, where the adversary loss is given by the sum of logistic loss of
        classification and weighted Lp-norm of the perturbation vectors. Do keep in mind that not all classifiers
        provide a well defined loss estimation function - therefore it is logistic loss, which is used instead.

        :param y_probas: Class-wise prediction probabilities.
        :param perturbations: Perturbations of samples towards being adversarial.
        :param targets: The target labels for the attack.
        :return: Aggregate loss score.
        """
        clf_loss_part = LowProFool.logistic_loss(y_probas, targets)
        norm_part = self.__weighted_lp_norm(perturbations)

        return clf_loss_part + self.lambd * norm_part

    def __apply_clipping(
            self,
            samples: np.ndarray,
            perturbations: np.ndarray
    ) -> np.ndarray:
        """
        Function for clipping perturbation vectors to forbid the adversary vectors to go beyond the allowed ranges of
        values.

        :param samples: Base design matrix.
        :param perturbations: Perturbations of samples towards being adversarial.
        :return: Clipped perturbation array.
        """
        if self.estimator.clip_values is None:
            return perturbations

        mins = self.estimator.clip_values[0]
        maxs = self.estimator.clip_values[1]

        np.clip(perturbations, mins - samples, maxs - samples, perturbations)
        return perturbations

    def __calculate_feature_importances(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> None:
        """
        This function calculates feature importances using a specified built-in function or applies a provided custom
        function (callable object). It calculates those values on the passed training data.

        :param x: Design matrix of the dataset used to train the classifier.
        :param y: Labels of the dataset used to train the classifier.
        :return: None.
        """
        if self.importance == 'pearson':
            # Apply a simple Pearson correlation calculation.
            pearson_correlations = [pearsonr(x[:, col], y)[0] for col in range(x.shape[1])]
            absolutes = np.abs(np.array(pearson_correlations))
            self.importance_vec = absolutes / np.power(np.sum(absolutes ** 2), 0.5)

        elif callable(self.importance):
            # Apply a custom function to call on the provided data.
            try:
                self.importance_vec = np.array(self.importance(x, y))
            except Exception as e:
                logger.exception("Provided importance function has failed.")
                raise e

            if not isinstance(self.importance_vec, np.ndarray):
                self.importance_vec = None
                raise TypeError("Feature importance vector should be of type np.ndarray or any convertible to that.")
            elif self.importance_vec.shape != (self.n_features,):
                self.importance_vec = None
                raise ValueError("Feature has to be one-dimensional array of size (n_features, ).")

        else:
            raise TypeError("Unrecognized feature importance function: {}".format(self.importance))

    def fit_importances(
            self,
            x: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            importance_array: Optional[np.ndarray] = None,
            normalize: Optional[bool] = True
    ):
        """
        This function allows one to easily calculate the feature importance vector using the pre-specified function,
        in case it wasn't passed at initialization.

        :param x: Design matrix of the dataset used to train the classifier.
        :param y: Labels of the dataset used to train the classifier.
        :param importance_array: Array providing features' importance score.
        :param normalize: Assure that feature importance values sum to 1.
        :return: LowProFool instance itself.
        """
        if not (importance_array is None):
            # Use a pre-calculated vector of feature importances.
            if np.array(importance_array).shape == (self.n_features, ):
                self.importance_vec = np.array(importance_array)
            else:
                raise ValueError("Feature has to be one-dimensional array of size (n_features, ).")

        elif self.importance_vec is None:
            # Apply a function specified at the LowProFool instance initialization.
            self.__calculate_feature_importances(np.array(x), np.array(y))

        if normalize:
            # Make sure that importance vector sums to 1.
            self.importance_vec = np.array(self.importance_vec) / np.sum(self.importance_vec)

        return self

    def generate(
            self,
            x: np.ndarray,
            y: Optional[np.ndarray] = None,
            **kwargs
    ) -> np.ndarray:
        """
        Generate adversaries for the samples passed in the `x` data matrix, whose targets are specified in `y`,
        one-hot-encoded target matrix. This procedure makes use of the LowProFool algorithm. In the case of failure,
        the resulting array will contain the initial samples on the problematic positions - which otherwise should
        contain the best adversary found in the process.

        :param x: An array with the original inputs to be attacked.
        :param y: One-hot-encoded target classes of shape (nb_samples, nb_classes).
        :param kwargs:
        :return: An array holding the adversarial examples.
        """
        if self.importance_vec is None:
            raise ValueError("No feature importance vector has been provided yet.")
        if y is None:
            raise ValueError("It is required to pass target classes as `y` parameter.")

        # Make sure samples and targets are of type np.ndarray.
        samples = np.array(x, dtype=np.float64)
        targets = np.array(y, dtype=np.float64)

        # Extract the target classes as integers implying their classes' indices.
        targets_integer = np.argmax(y, axis=1)

        if targets.shape[1] != self.n_classes:
            raise ValueError("Targets shape is not compatible with number of classes.")
        if samples.shape[1] != self.n_features:
            raise ValueError("Samples shape is not compatible with number of features.")

        # Initialize perturbation vectors and learning rate.
        perturbations = np.zeros(samples.shape, dtype=np.float64)
        eta = self.eta

        # Initialize 'keep-the-best' variables.
        best_norm_losses = np.inf * np.ones(samples.shape[0], dtype=np.float64)
        best_perturbations = perturbations.copy()

        # Calculate class-wise probabilities.
        y_probas = self.estimator.predict((samples + perturbations).astype(np.float32))
        success_indicators = np.zeros(samples.shape[0], dtype=np.float64)

        # Predicate used to determine whether the target was met based on the given probabilities
        def met_target(y_probas, target_class):
            if self.threshold is None:
                return np.argmax(y_probas) == target_class
            else:
                return y_probas[target_class] > self.threshold

        # Main loop.
        for i in trange(self.n_steps, desc="LowProFool", disable=not(self.verbose)):
            # Calculate gradients, apply them to perturbations and clip if needed.
            grad = self.__get_gradients(samples, perturbations, y_probas, targets)
            perturbations -= eta * grad
            perturbations = self.__apply_clipping(samples, perturbations)

            # Decrease learning rate for the next iteration.
            eta = max(eta * self.eta_decay, self.eta_min)

            # Calculate class-wise probabilities.
            y_probas = self.estimator.predict((samples + perturbations).astype(np.float32))

            # Examine the quality of adversaries in the current step.
            for j, target_int in enumerate(targets_integer):
                # Check for every sample whether the threshold probability is reached.
                if met_target(y_probas[j], target_int):
                    success_indicators[j] = 1.
                    # Calculate weighted Lp-norm loss.
                    norm_loss = self.__weighted_lp_norm(perturbations[j:j+1])[0, 0]

                    # Note it, if the adversary improves.
                    if norm_loss < best_norm_losses[j]:
                        best_norm_losses[j] = norm_loss
                        best_perturbations[j] = perturbations[j].copy()

        logger.info("Success rate of LowProFool attack: {:.2f}%".format(
            100 * np.sum(success_indicators) / success_indicators.size
        ))

        # The generated adversaries are a sum of initial samples and best perturbation vectors found by the algorithm.
        return samples + best_perturbations

    def _check_params(self) -> None:
        """
        Check correctness of parameters.

        :return: None.
        """
        if not(isinstance(self.n_classes, int) and self.n_classes > 0):
            raise ValueError('The argument `n_classes` has to be positive integer.')

        if not(isinstance(self.n_features, int) and self.n_classes > 0):
            raise ValueError('The argument `n_features` has to be positive integer.')

        if not(isinstance(self.n_steps, int) and self.n_steps > 0):
            raise ValueError('The argument `n_steps` (number of iterations) has to be positive integer.')

        if not(
            (isinstance(self.threshold, float) and 0 < self.threshold < 1) or
            self.threshold is None
        ):
            raise ValueError('The argument `threshold` has to be either float in range (0, 1) or None.')

        if not(isinstance(self.lambd, (float, int)) and self.lambd >= 0):
            raise ValueError('The argument `lambd` has to be non-negative float or integer.')

        if not(isinstance(self.eta, (float, int)) and self.eta > 0):
            raise ValueError('The argument `eta` has to be positive float or integer.')

        if not(isinstance(self.eta_decay, (float, int)) and 0 < self.eta_decay <= 1):
            raise ValueError('The argument `eta_decay` has to be float or integer in range (0, 1].')

        if not(isinstance(self.eta_min, (float, int)) and self.eta_min >= 0):
            raise ValueError('The argument `eta_min` has to be non-negative float or integer.')

        if not(
            (isinstance(self.p, (float, int)) and self.p > 0) or
            (isinstance(self.p, str) and self.p == "inf") or
            self.p == np.inf
        ):
            raise ValueError('The argument `p` has to be either positive-valued float or integer, np.inf, or "inf".')

        if not(
            isinstance(self.importance, (str, Callable)) or
            (isinstance(self.importance, np.ndarray) and self.importance.shape == (self.n_features, ))
        ):
            raise ValueError('The argument `importance` has to be either string, ' +
                             'callable or np.ndarray of the shape (n_features, ).')

        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')
