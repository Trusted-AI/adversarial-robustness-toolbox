# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
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
# Copyright (c) 2022 Minhao Cheng
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
This module implements the Sign-OPT attack `SignOPTAttack`. This is a query-efficient
hard-label adversarial attack.
| Paper link: https://arxiv.org/pdf/1909.10773.pdf
"""

import logging
from typing import Optional, TYPE_CHECKING, Tuple
import time
import numpy as np
from numpy import integer, linalg as LA
from tqdm.auto import tqdm

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class SignOPTAttack(EvasionAttack):
    """
    Implements the Sign-OPT attack `SignOPTAttack`. This is a query-efficient
    hard-label adversarial attack.
    | Paper link: https://arxiv.org/pdf/1909.10773.pdf
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "epsilon",
        "num_trial",
        "max_iter",
        "query_limit",
        "K",
        "alpha",
        "beta",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        targeted: bool = True,
        epsilon: float = 0.001,
        num_trial: int = 100,
        max_iter: int = 1000,  # recommend 5000 for targeted attack
        query_limit: int = 20000,  # recommend 40000 for targeted attack
        k: int = 200,
        alpha: float = 0.2,
        beta: float = 0.001,
        eval_perform: bool = False,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> None:
        """
        Create a Sign_OPT attack instance.
        :param estimator: A trained classifier.
        :param targeted: Should the attack target one specific class.
        :param epsilon: A very small smoothing parameter.
        :param num_trial: A number of trials to calculate a good starting point
        :param max_iter: Maximum number of iterations
        :param query_limit: Limitation for number of queries to prediction model
        :param k: Number of random directions (for estimating the gradient)
        :param alpha: The step length for line search
        :param beta: The tolerance for line search
        :param batch_size: The size of the batch used by the estimator during inference.
        :param verbose: Show detailed information
        :param eval_perform: Evaluate performnace with Avg. L2 and Success Rate with randomly choosing 100 samples
        """

        super().__init__(estimator=estimator)
        self.targeted = targeted
        self.epsilon = epsilon
        self.num_trial = num_trial
        self.max_iter = max_iter
        self.query_limit = query_limit

        self.k = k
        self.alpha = alpha
        self.beta = beta

        self.batch_size = batch_size
        self.verbose = verbose

        self.eval_perform = eval_perform
        if eval_perform:
            self.logs = np.zeros(100)
        # self.clip_min: Union[float]
        # self.clip_max: Union[float]
        if self.estimator.clip_values is not None:
            self.clip_min, self.clip_max = self.estimator.clip_values
            self.enable_clipped = True
        else:
            # self.clip_min, self.clip_max = 0.0, 1.0
            self.enable_clipped = False
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of
                        shape (nb_samples, nb_classes) or indices of shape
                        (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :return: An array holding the adversarial examples.
        """
        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        targets = check_and_transform_label_format(y, self.estimator.nb_classes, return_one_hot=False)

        if targets is not None and self.estimator.nb_classes == 2 and targets.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Assert that if attack is targeted, targets is provided
        if self.targeted and targets is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")
        # Assert that if attack is targeted, training data is provided
        if self.targeted:
            x_init = kwargs["x_init"]
            if x_init is None:
                raise ValueError("`x_init` needs to be provided for a targeted attack.")

        # Get clip_min and clip_max infer them from data, otherwise, it is initialized by self.estimator
        if self.clip_min is None and self.clip_max is None:
            self.clip_min, self.clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Generate the adversarial samples
        counter = 0  # only do the performance tests with 100 samples
        for ind, val in enumerate(tqdm(x_adv, desc="Sign_OPT attack", disable=not self.verbose)):
            if self.targeted:
                if targets[ind] == preds[ind]:
                    if self.verbose:
                        print("Image already targeted. No need to attack.")
                    continue

                x_adv[ind], diff, succeed = self._attack(  # diff and succeed are for performance test
                    x_0=val,
                    y_0=preds[ind],
                    target=targets[ind],
                    x_init=x_init,
                )
            else:
                x_adv[ind], diff, succeed = self._attack(  # diff and succeed are for performance test
                    x_0=val,
                    y_0=preds[ind],
                )
            if succeed and self.eval_perform and counter < 100:
                self.logs[counter] = LA.norm(diff)
                counter += 1

        # todo: the compute_success() doesn't work for targeted case, dimension related error
        if self.targeted is False:
            logger.info(
                "Success rate of Sign_OPT attack: %.2f%%",
                # 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
                100 * compute_success(self.estimator, x, targets, x_adv, self.targeted),
            )

        return x_adv  # all images with untargeted adversarial

    def _fine_grained_binary_search(self, x_0, y_0, theta, initial_lbd, current_best, target: Optional[int] = None):
        if self.targeted:
            tolerate = 1e-5
            y_0 = target
        else:
            tolerate = 1e-3
        nquery = 0
        if initial_lbd > current_best:
            # the condition in "argmin()" in page 3, objective function 1
            # if the adv != y0, in case of
            # target: failed to make adv to target
            # untarget: failed to make adv to something else
            pred = self._is_label(x_0 + current_best * theta, y_0)
            if self.targeted != pred:
                # if targeted, pred should be True
                # if not targeted, pred should be False
                # in case of NOT satisfy above condition, return
                nquery += 1
                return float("inf"), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > tolerate:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if not self._is_label(x_0 + lbd_mid * theta, y_0):
                if self.targeted:
                    lbd_lo = lbd_mid
                else:
                    lbd_hi = lbd_mid
            else:
                if self.targeted:
                    lbd_hi = lbd_mid
                else:
                    lbd_lo = lbd_mid
        return lbd_hi, nquery

    # perform the line search in paper (Chen and Zhang, 2019)
    # paper link: https://openreview.net/pdf?id=rJlk6iRqKX
    def _fine_grained_binary_search_local(self, x_0, y_0, target, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        # For targeted: we want to expand(x1.01) boundary away from targeted dataset
        #     prediction(x0+lbd*theta) != target, GOOD
        # For untargeted, we want to slim(x0.99) the boundary toward the orginal dataset
        #     prediction(x0+lbd*theta) == original, GOOD
        if self.targeted:
            y_0 = target

        if self._is_label(x_0 + lbd * theta, y_0) != self.targeted:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self._is_label(x_0 + lbd_hi * theta, y_0) != self.targeted:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float("inf"), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self._is_label(x_0 + lbd_lo * theta, y_0) == self.targeted:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self._is_label(x_0 + lbd_mid * theta, y_0) == self.targeted:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    # temp method if ART has a similar method
    # x0: dimension is [1, 28, 28]
    # return True, if prediction of x0 is label, False otherwise
    def _is_label(self, x_0, label) -> bool:
        if self.enable_clipped:
            x_0 = np.clip(x_0, self.clip_min, self.clip_max)
        pred = self.estimator.predict(np.expand_dims(x_0, axis=0), batch_size=self.batch_size)
        pred_y0 = np.argmax(pred)
        return pred_y0 == label

    # temp method if ART has a similar method
    # x0: dimension is [1, 28, 28]
    # return predicted label
    def _predict_label(self, x_0) -> integer:
        if self.enable_clipped:
            x_0 = np.clip(x_0, self.clip_min, self.clip_max)
        pred = self.estimator.predict(np.expand_dims(x_0, axis=0), batch_size=self.batch_size)
        return np.argmax(pred)

    def _sign_grad(self, x_0, y_0, epsilon, theta, initial_lbd, target=None):
        # Evaluate the sign of gradient by formul
        sign_grad = np.zeros(theta.shape).astype(np.float32)
        queries = 0
        # use orthogonal transform
        for _ in range(self.k):  # for each u
            # Algorithm 1: Sign-OPT attack
            #     A:Randomly sample u1, . . . , uQ from a Gaussian or Uniform distribution;
            u_g = np.random.randn(*theta.shape).astype(np.float32)
            # gaussian
            u_g /= LA.norm(u_g)
            # function (3) in the paper
            new_theta = theta + epsilon * u_g
            new_theta /= LA.norm(new_theta)
            sign = 1
            if self.targeted:
                y_0 = target
            # Untargeted case
            if self.targeted == self._is_label(x_0 + initial_lbd * new_theta, y_0):
                sign = -1

            queries += 1
            # Algorithm 1: Sign-OPT attack
            #     B:Compute gˆ ←
            sign_grad += u_g * sign

        sign_grad /= self.k

        return sign_grad, queries

    def _attack(
        self,
        x_0: np.ndarray,
        y_0: int,
        target: Optional[int] = None,  # for targeted attack
        x_init: Optional[np.ndarray] = None,  # for targeted attack
        distortion=None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        query_count = 0
        ls_total = 0

        # init: Calculate a good starting point (direction)
        num_directions = self.num_trial
        best_theta, g_theta = np.zeros((0, 0)), float("inf")
        if self.verbose:
            print(f"Searching for the initial direction on {num_directions} random directions: ")
        if self.targeted and x_init is not None:
            if self.verbose:
                print(f"this is targeted attack, org_label={y_0}, target={target}")
            sample_count = 0
            for i, x_i in enumerate(x_init):
                # yi_pred = model.predict_label(xi.cuda())
                # find a training data which label is target
                yi_pred = self._predict_label(x_i)
                query_count += 1
                if yi_pred != target:
                    continue

                theta = x_i - x_0  # compared to example, xi.cup().numpy() and x0.cpu().numpy() are removed.
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self._fine_grained_binary_search(x_0, y_0, theta, initial_lbd, g_theta, target)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                sample_count += 1
                if sample_count >= self.num_trial or i > 500:
                    break
        else:
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x_0.shape).astype(np.float32)  # gaussian distortion
                # register adv directions
                # if self._is_label(x0 + theta, y0) == False:
                if not self._is_label(x_0 + theta, y_0):
                    initial_lbd = LA.norm(theta)
                    theta /= initial_lbd  # l2 normalize: theta is normalized
                    # getting smaller g_theta
                    lbd, count = self._fine_grained_binary_search(x_0, y_0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        if self.verbose:
                            print(f"Found distortion {g_theta} with iteration/num_directions={i}/{num_directions}")

        # fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float("inf"):
            if self.verbose:
                print("Couldn't find valid initial, failed")
            return (
                x_0,
                np.zeros((0, 0)),
                False,
            )  # , query_count, best_theta # test data, ?, ?, # of queries, best_theta(Gaussian L2 norm)

        query_limit = self.query_limit
        alpha = self.alpha
        beta = self.beta
        timestart = time.time()
        # Begin Sign_OPT from here
        # Algorithm 1: Sign-OPT attack
        #     A:Randomly sample u1, . . . , uQ from a Gaussian or Uniform distribution;
        #     B:Compute gˆ ←  ...
        #     C:Update θt+1 ← θt − ηgˆ;
        #     D:Evaluate g(θt) using the same search algorithm in
        #       Cheng et al. (2019) https://openreview.net/pdf?id=rJlk6iRqKX,
        x_g, g_g = best_theta, g_theta
        distortions = [g_g]
        iterations = self.max_iter
        for i in range(iterations):
            sign_gradient, grad_queries = self._sign_grad(x_0, y_0, self.epsilon, x_g, g_g, target)

            # Line search of the step size of gradient descent
            ls_count = 0
            min_theta = x_g  # next theta
            min_g2 = g_g  # current g_theta
            new_theta = np.zeros((0, 0))
            for _ in range(15):
                # Algorithm 1: Sign-OPT attack
                #     C:Update θt+1 ← θt − ηgˆ;
                new_theta = x_g - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                # Algorithm 1: Sign-OPT attackx
                #     D:Evaluate g(θt) using the same search algorithm in
                #       Cheng et al. (2019) https://openreview.net/pdf?id=rJlk6iRqKX,
                #       **Algorithm 1 Compute g(θ) locally**
                new_g2, count = self._fine_grained_binary_search_local(
                    x_0, y_0, target, new_theta, initial_lbd=min_g2, tol=beta / 500
                )
                ls_count += count
                alpha = alpha * 2  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break  # meaning alphia is too big, so it needs to be reduced.

            if min_g2 >= g_g:  # if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = x_g - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self._fine_grained_binary_search_local(
                        x_0, y_0, target, new_theta, initial_lbd=min_g2, tol=beta / 500
                    )
                    ls_count += count
                    if new_g2 < g_g:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if alpha < 1e-4:  # if the above two blocks of code failed
                alpha = 1.0
                if self.verbose:
                    print("Warning: not moving")
                beta = beta * 0.1
                if beta < 1e-8:
                    break

            # if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            x_g, g_g = min_theta, min_g2

            query_count += grad_queries + ls_count
            ls_total += ls_count
            distortions.append(g_g)

            if query_count > query_limit:
                if self.verbose:
                    print(f"query_count={query_count} > query_limit={query_limit}")
                break

            if self.verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1} distortion  {g_g} num_queries {query_count}")
        timeend = time.time()
        succeed = False
        if self.targeted is False and (distortion is None or g_g < distortion):
            succeed = True
            if self.verbose:
                print(
                    f"Succeed distortion {g_g} org_label {y_0} predict_lable {target} \
                    queries {query_count} Line Search queries {ls_total}"
                )
                target_pred = self._predict_label(x_0 + g_g * x_g)
                if target_pred == y_0:
                    print(f"WARNING: prediction on adv {target_pred} == org label {y_0}")
            # return self._clip_value(x_0 + g_g * x_g), g_g * x_g, True
        elif self.targeted and self._is_label(x_0 + g_g * x_g, target):
            succeed = True
            if self.verbose:
                print(
                    f"Adversarial Example Found Successfully: distortion {g_g} target, \
                    {target} queries {query_count} Line Search queries {ls_total} Time: {timeend-timestart} seconds"
                )
            # return self._clip_value(x_0 + g_g * x_g), g_g * x_g, True
        else:
            succeed = False
            if self.verbose:
                print(f"Failed: distortion {g_g}")
        return self._clip_value(x_0 + g_g * x_g), g_g * x_g, succeed

    def _clip_value(self, x_0) -> np.ndarray:
        if self.enable_clipped:
            x_0 = np.clip(x_0, self.clip_min, self.clip_max)
        return x_0

    def _check_params(self) -> None:
        if not isinstance(self.targeted, bool):
            raise ValueError("The argument `targeted` has to be of type bool.")

        if self.epsilon <= 0:
            raise ValueError("The initial step size for the step towards the target must be positive.")

        if not isinstance(self.num_trial, int) or self.num_trial < 0:
            raise ValueError("The number of trials must be a non-negative integer.")

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.query_limit, int) or self.query_limit <= 0:
            raise ValueError("The number of query_limit must be a positive integer.")

        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(
                "The number of random directions (for estimating the gradient) must be a positive integer."
            )

        if self.alpha <= 0:
            raise ValueError("The value of alpha must be positive.")

        if self.beta <= 0:
            raise ValueError("The value of beta must be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
