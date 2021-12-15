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
        # Todo: make the dimentions to be compatible to ART, use batch_size???
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
            # 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted),
        )
        
        return x_adv # all images with untargeted adversarial
    
    def _fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self._is_org_single_image(x0+current_best*theta, y0):
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
            if self._is_org_single_image(x0+lbd_mid*theta, y0) == False:    
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
    
    def _fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        # still inside boundary
        if self._is_org_single_image(x0+lbd*theta, y0):
        # if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            # self._is_org_single_image(x0+current_best*theta, y0):
            while self._is_org_single_image(x0+lbd_hi*theta, y0):
            # while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self._is_org_single_image(x0+lbd_lo*theta, y0) == False:
            # while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self._is_org_single_image(x0+lbd_mid*theta, y0) == False:
            # if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        # print(f'In _fine_grained_binary_search_local(), with initial_lbd={initial_lbd} returning lbd_hi={lbd_hi}, nquery={nquery}')
        return lbd_hi, nquery
    
    # temp method
    # x0: dimension is [1, 28, 28]
    # org_y0: type of ...
    # return True, if prediction of x0 is org_y0, False otherwise
    def _is_org_single_image(self, x0, org_y0, verbose=False) -> bool:
        pred = self.estimator.predict(np.expand_dims(x0, axis=0))
        pred_y0 = np.argmax(pred)
        if verbose:
            print(f'pred_lable={pred_y0}, orginal_label={org_y0}')
        return pred_y0 == org_y0
    
    def _predict_single_image(self, x0, org_y0=None, verbose=False) -> bool:
        pred = self.estimator.predict(np.expand_dims(x0, axis=0))
        return np.argmax(pred)
        
    
    def _sign_grad_v1(self, x0, y0, theta, initial_lbd):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = 200 # 200 random directions (for estimating the gradient)
        h = 0.001 # todo: can be passed in as a parameter
        sign_grad = np.zeros(theta.shape).astype(np.float32)
        queries = 0
        ### USe orthogonal transform
        for iii in range(K): # for each u
            u = np.random.randn(*theta.shape).astype(np.float32); u /= LA.norm(u)
            new_theta = theta + h*u; new_theta /= LA.norm(new_theta)
            sign = 1

            # Untargeted case
            # preds.append(self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()).item())
            # if (self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) != y0): # success
            if self._is_org_single_image(x0+initial_lbd*new_theta, y0) == False:    
                sign = -1

            queries += 1
            sign_grad += u*sign
        
        sign_grad /= K
        
        return sign_grad, queries

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
        distortion = None
        #### init: Calculate a good starting point (direction)
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape).astype(np.float32) # gaussian distortion
            # register adv directions
            if self._is_org_single_image(x0+theta, y0) == False:
                # print(f"iteration/num_directions={i}/{num_directions}")
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd # l2 normalize: theta is normalized
                lbd, count = self._fine_grained_binary_search(x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)
                    print(f"iteration/num_directions={i}/{num_directions}")
        timeend = time.time()
        print(f'Spent {timeend-timestart} seconds for finding directions')
        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf') or g_theta == np.inf: ## todo: why two types?
            print("Couldn't find valid initial, failed")
            return 
        ## todo: consider to pass following variables as parameters 
        momentum = 0.0
        query_limit = 20000
        alpha = 0.2
        beta = 0.001
        #### Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        distortions = [gg]
        iterations = 1000
        for i in range(iterations):
            ## gradient estimation at x0 + theta (init)
            sign_gradient, grad_queries = self._sign_grad_v1(x0, y0, xg, initial_lbd=gg)
            
            ## Line search of the step size of gradient descent
            ls_count = 0
            min_theta = xg ## next theta
            min_g2 = gg ## current g_theta
            min_vg = vg ## velocity (for momentum only)
            for _ in range(15):
                # update theta by one step sgd
                if momentum > 0:
                    new_vg = momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)

                new_g2, count = self._fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2 # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break

            if min_g2 >= gg: ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
                        new_vg = momentum*vg - alpha*sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self._fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        if momentum > 0:
                            min_vg = new_vg
                        break

            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2
            vg = min_vg

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)
            
            if query_count > query_limit:
                print(f'query_count={query_count} > query_limit={query_limit}')
                break
            
            ## logging
            if (i + 1) % 10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
            # self.log[i+1][0], self.log[i+1][1] = gg, query_count
            # if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break
        
        if distortion is None or gg < distortion:
            target = self._predict_single_image(x0 + gg*xg, y0)
            print("Succeed distortion {:.4f} org {:d} target"
                  " {:d} queries {:d} LS queries {:d}\n".format(gg, y0, target, query_count, ls_total))
            # return x0 + gg*xg, gg, True, query_count, xg
            return x0 + gg*xg
        
        timeend = time.time()
        print("\nFailed: distortion %.4f" % (gg))
        
        # self.log[i+1:,0] = gg
        # self.log[i+1:,1] = query_count
        
        # return x0 + gg*xg, gg, False, query_count, xg
        return x0 + gg*xg
        
    
    def _check_params(self) -> None:
        # Todo: add other param checking
        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
    
