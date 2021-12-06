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
    """

    Args:
        EvasionAttack ([type]): [description]
    | Paper link: https://arxiv.org/pdf/1909.10773.pdf
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
        # num_trial: int = 100,
        # iterations: int = 1000,
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
        # self.num_trial = num_trial
        # self.iteration = iterations

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
        
        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)
        
        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="Sign_OPT attack", disable=not self.verbose)):
            if self.targeted:
                print("Not implemented")
                return x_adv
            else:
                x_adv[ind] = self._attack( # one image
                    x0=val,
                    y0=preds[ind],
                )
            
        logger.info(
            "Success rate of Sign_OPT attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )
        
        return x_adv # all images with untargeted adversarial
    
    def _fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+torch.tensor(current_best*theta, dtype=torch.float).cuda()) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        
        while (lbd_hi - lbd_lo) > 1e-3: # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def _attack(
        self,
        x0: np.ndarray,
        # y: int, # for targeted attack
        y0: int,
    ):
        """
        Algorithm 1: Sign-OPT attack
            Randomly sample u1, . . . , uQ from a Gaussian or Uniform distribution;
            Compute gˆ ←  ...
            Update θt+1 ← θt − ηgˆ;
            Evaluate g(θt) using the same search algorithm in Cheng et al. (2019) https://openreview.net/pdf?id=rJlk6iRqKX, **Algorithm 1 Compute g(θ) locally**
        """
        query_count = 0
        ls_total = 0
        #### init: Calculate a good starting point (direction)
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape).astype(np.float32) # gaussian distortion
            # register adv directions
            pred_cur = self.estimator.predict(np.expand_dims(x0+theta, axis=0))
            if np.argmax(pred_cur) != y0: 
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd # l2 normalize
                lbd, count = self._fine_grained_binary_search(self.estimator, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)
        ## fail if cannot find a adv direction within 200 Gaussian
        #### Begin Gradient Descent.
            ## gradient estimation at x0 + theta (init)
            ## Line search of the step size of gradient descent
            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
        # x0 + distortion
        # return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), gg, False, query_count, xg
        return x0
    
    def _check_params(self) -> None:
        # Todo: add other param checking
        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
    
