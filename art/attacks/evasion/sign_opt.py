# todo: license

"""
todo: summary 
"""

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy import linalg as LA
from tqdm.auto import tqdm, trange
import torch
import time

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
    
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self, 
        estimator: "CLASSIFIER_TYPE", 
        targeted: bool = True,
        num_trial: int = 100,
        iterations: int = 1000,
        verbose: bool = True,) -> None:
        """
        Create a Sign_OPT attack instance.

        :param estimator: A trained classifier.
        :param targeted: Should the attack target one specific class.
        :param verbose: Show progress bars.
        """
        
        super().__init__(estimator=estimator)
        self._targeted = targeted
        self.num_trial = num_trial
        self.iteration = iterations

        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes, return_one_hot=False)
        
        if y is not None and self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )
            
        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x), axis=1)
        # (10000, 1, 28, 28) (10000,)
        # print(x.shape, preds.shape)
        
        
        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)
        
        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="Sign_OPT attack", disable=not self.verbose)):
            if self.targeted:
                print("Not implemented")
                return x_adv
            else:
                x_adv[ind] = self._attack(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                )
            
        logger.info(
            "Success rate of Sign_OPT attack: %.2f%%",
            # 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted),
        )
        
        return x_adv
    
    def _attack(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
    ):
        query_count = 0
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (self.num_trial))
        timestart = time.time()
        
        for i in range(self.num_trial):
            query_count += 1
            theta = np.random.randn(*x.shape) # gaussian distortion
            # register adv directions
            if self.estimator.predict(x+torch.tensor(theta, dtype=torch.float).cuda()) != y:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd # l2 normalize
                lbd, count = self.fine_grained_binary_search(self.estimator, x, y, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)
        if g_theta == np.inf:
            return "NA", float('inf'), 0
        
        timeend = time.time()
        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x, 0, False, query_count, best_theta
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend-timestart, query_count))
        self.log[0][0], self.log[0][1] = g_theta, query_count
    
        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        stopping = 0.005
        prev_obj = 100000
        for i in range(self.iterations):
            gradient = np.zeros(theta.shape)
            q = 5
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u.flatten(),np.inf)
                ttt = theta+beta * u
                ttt /= LA.norm(ttt.flatten(),np.inf)
                g1, count = self.fine_grained_binary_search_local(self.estimator, x, y, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient

            if (i+1)%1 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, LA.norm((g2*theta).flatten(),np.inf), opt_count))
                if g2 > prev_obj-stopping:
                    print("stopping")
                    break
                prev_obj = g2

            min_theta = theta
            min_g2 = g2
        
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta.flatten(),np.inf)
                new_g2, count = self.fine_grained_binary_search_local(self.estimator, x, y, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta.flatten(),np.inf)
                    new_g2, count = self.fine_grained_binary_search_local(self.estimator, x, y, new_theta, initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2
            
            #print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.00005):
                    break

        target = self.estimator.predict(x0 + g_theta*best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        return x + g_theta*best_theta, g_theta, query_count + opt_count
    
    def _check_params(self) -> None:
        # todo: add param checking
        return super()._check_params()