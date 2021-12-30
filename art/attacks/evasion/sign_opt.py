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
        "alpha"
        "delta",
        "epsilon",
        "max_iter",
        "num_trial",
    ]
    
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self, 
        estimator: "CLASSIFIER_TYPE", 
        targeted: bool = True,
        epsilon: int = 0.001,
        num_trial: int = 100,
        max_iter: int = 1000, # todo 5000 for targeted?
        query_limit = 20000, # todo 40000 for targeted?
        K = 200,
        alpha = 0.2,
        beta = 0.001,
        verbose: bool = True,
        ) -> None:
        """
        Create a Sign_OPT attack instance.

        :param estimator: A trained classifier.
        :param targeted: Should the attack target one specific class.
        :param verbose: Show progress bars.
        """
        
        super().__init__(estimator=estimator)
        self.targeted = targeted
        self.epsilon = epsilon
        self.num_trial = num_trial
        self.max_iter = max_iter
        self.query_limit = query_limit
        
        self.K = K
        self.alpha = alpha
        self.beta = beta

        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, targets: Optional[np.ndarray] = None, x_train: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param targets: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :param x_train: Training Dataset, todo: do we really need this?
        :return: An array holding the adversarial examples.
        """
        
        targets = check_and_transform_label_format(targets, self.estimator.nb_classes, return_one_hot=False)
        
        if targets is not None and self.estimator.nb_classes == 2 and targets.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )
            
        # Assert that, if attack is targeted, targets is provided
        if self.targeted and targets is None:  
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")
        # Assert that, if attack is targeted, training data is provided
        if self.targeted and x_train is None:  
            raise ValueError("Training Data `x_train` needs to be provided for a targeted attack.")

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x), axis=1)
        
        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)
        
        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="Sign_OPT attack", disable=not self.verbose)):
            if self.targeted:
                if targets[ind] == preds[ind]:
                    print("Image already targeted. No need to attack.")
                    continue

                x_adv[ind] = self._attack( # one image
                    x0=val,
                    y0=preds[ind],
                    target=targets[ind],
                    x_train=x_train,
                )
            else:
                x_adv[ind] = self._attack( # one image
                    x0=val,
                    y0=preds[ind],
                )
            
        # todo: the compute_success() doesn't work for targeted case, dimension related error
        if self.targeted == False:
            logger.info(
                "Success rate of Sign_OPT attack: %.2f%%",
                # 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
                100 * compute_success(self.estimator, x, targets, x_adv, self.targeted),
            )
        
        return x_adv # all images with untargeted adversarial
    
    ## It is used at the beginning, finding a good start direction
    def _fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best, target: Optional[int] = None):
        if self.targeted:
            tolerate = 1e-5 
            y0 = target
        else:
            tolerate = 1e-3
        nquery = 0
        if initial_lbd > current_best: 
            # the condition in "argmin()" in page 3, objective function 1
            pred = self._is_label(x0+current_best*theta, y0)
            if self.targeted != pred:
                # if targeted, pred should be True
                # if not targeted, pred should be False
                # in case of NOT satisfy above condition, return
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        
        while (lbd_hi - lbd_lo) > tolerate: 
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self._is_label(x0+lbd_mid*theta, y0) == False:    
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        # Chloe - workaround to avoid lbd_hi keeps decreasing because the image combines with distorsion always gives False after qurey above: self._is_label(x0+lbd_mid*theta, y0) 
        print(f'lbd_hi={lbd_hi}, current_best={current_best}, initial_lbd={initial_lbd}')
        if lbd_lo == 0.0:
            return float('inf'), nquery
        return lbd_hi, nquery
    
    # perform the line search in paper 2019
    # https://openreview.net/pdf?id=rJlk6iRqKX
    def _fine_grained_binary_search_local(self, x0, y0, target, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        """
        For targeted: we want to expand(x1.01) boundary away from targeted dataset
            prediction(x0+lbd*theta) != target, GOOD
        For untargeted, we want to slim(x0.99) the boundary toward the orginal dataset
            prediction(x0+lbd*theta) == original, GOOD
        """
        if self.targeted: y0 = target
        # still inside boundary
        if self._is_label(x0+lbd*theta, y0) != self.targeted:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self._is_label(x0+lbd_hi*theta, y0)!= self.targeted:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self._is_label(x0+lbd_lo*theta, y0) == self.targeted:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self._is_label(x0+lbd_mid*theta, y0) == self.targeted:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        # print(f'In _fine_grained_binary_search_local(), with initial_lbd={initial_lbd} returning lbd_hi={lbd_hi}, nquery={nquery}')
        return lbd_hi, nquery
    
    # temp method
    # x0: dimension is [1, 28, 28]
    # org_y0: type of ...
    # return True, if prediction of x0 is org_y0, False otherwise
    def _is_label(self, x0, label, verbose=False) -> bool:
        pred = self.estimator.predict(np.expand_dims(x0, axis=0))
        pred_y0 = np.argmax(pred)
        if verbose:
            print(f'pred_lable={pred_y0}, orginal_label={label}')
        return pred_y0 == label
    
    def _predict_label(self, x0, org_y0=None, verbose=False) -> bool:
        pred = self.estimator.predict(np.expand_dims(x0, axis=0))
        return np.argmax(pred)
        
    
    def _sign_grad(self, x0, y0, epsilon, theta, initial_lbd, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        # todo: put k as class varible
        K = self.K # 200 random directions (for estimating the gradient)
        sign_grad = np.zeros(theta.shape).astype(np.float32)
        queries = 0
        ### use orthogonal transform
        for iii in range(K): # for each u
            """
            Algorithm 1: Sign-OPT attack
                A:Randomly sample u1, . . . , uQ from a Gaussian or Uniform distribution; 
            """
            u = np.random.randn(*theta.shape).astype(np.float32); # gaussian
            u /= LA.norm(u)
            # function (3) in the paper
            new_theta = theta + epsilon*u; 
            new_theta /= LA.norm(new_theta)
            sign = 1
            if self.targeted: y0 = target
            # Untargeted case
            if self.targeted == self._is_label(x0+initial_lbd*new_theta, y0):   
                sign = -1
            # elif self.targeted and self._is_label(x0+initial_lbd*new_theta, target):
            #     sign = -1
            # else:
            #     print('WARNING: unexpected combination of conditions')

            queries += 1
            """
            Algorithm 1: Sign-OPT attack
                B:Compute gˆ ←  ...
            """
            sign_grad += u*sign
        
        sign_grad /= K
        
        return sign_grad, queries

    def _attack(
        self,
        x0: np.ndarray,
        y0: int,
        target: Optional[int]=None, # for targeted attack
        x_train: Optional[np.ndarray]=None, # for targeted attack
        distortion = None, 
    ) -> np.ndarray:
        query_count = 0
        ls_total = 0
    
        ### init: Calculate a good starting point (direction)
        num_directions = self.num_trial
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        if self.targeted:
            print(f'this is targeted attack, org_label={y0}, target={target}')
            sample_count = 0 
            for i, xi in enumerate(x_train):
                # yi_pred = model.predict_label(xi.cuda())
                # find a training data which label is target
                yi_pred = self._predict_label(xi)
                query_count += 1
                if yi_pred != target:
                    continue
                
                theta = xi - x0 # compared to example, xi.cup().numpy() and x0.cpu().numpy() are removed. 
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self._fine_grained_binary_search(x0, y0, theta, initial_lbd, g_theta, target)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    # print("--------> Found distortion %.4f" % g_theta)
                    print(f"Found distortion {g_theta} with sample_count-iteration-queryCnt={sample_count}-{i}-{query_count}")
                sample_count += 1
                if sample_count >= self.num_trial or i > 500:
                    break
        else:
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x0.shape).astype(np.float32) # gaussian distortion
                # register adv directions
                if self._is_label(x0+theta, y0) == False:
                    initial_lbd = LA.norm(theta)
                    theta /= initial_lbd # l2 normalize: theta is normalized
                    # getting smaller g_theta
                    lbd, count = self._fine_grained_binary_search(x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print(f"Found distortion {g_theta} with iteration/num_directions={i}/{num_directions}")
        
        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'): 
            print("Couldn't find valid initial, failed")
            return x0, 0, False, query_count, best_theta # test data, ?, ?, # of queries, best_theta(Gaussian L2 norm)
        
        query_limit = self.query_limit 
        alpha = self.alpha
        beta = self.beta
        timestart = time.time()
        ### Begin Sign_OPT from here 
        """
        Algorithm 1: Sign-OPT attack
            A:Randomly sample u1, . . . , uQ from a Gaussian or Uniform distribution; 
            B:Compute gˆ ←  ...
            C:Update θt+1 ← θt − ηgˆ;
            D:Evaluate g(θt) using the same search algorithm in Cheng et al. (2019) https://openreview.net/pdf?id=rJlk6iRqKX, 
        """
        xg, gg = best_theta, g_theta
        distortions = [gg]
        iterations = self.max_iter
        for i in range(iterations): 
            sign_gradient, grad_queries = self._sign_grad(x0, y0, self.epsilon, xg, gg, target)
            
            ## Line search of the step size of gradient descent
            ls_count = 0
            min_theta = xg ## next theta
            min_g2 = gg ## current g_theta
            for _ in range(15): # why 15? 15 is the region?
                # print('^',end=' ')
                """
                Algorithm 1: Sign-OPT attack
                    C:Update θt+1 ← θt − ηgˆ;
                """
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                """
                Algorithm 1: Sign-OPT attackx
                    D:Evaluate g(θt) using the same search algorithm in Cheng et al. (2019) https://openreview.net/pdf?id=rJlk6iRqKX, **Algorithm 1 Compute g(θ) locally**
                """
                new_g2, count = self._fine_grained_binary_search_local(x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2 # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break # meaning alphia is too big, so it needs to be reduced. 

            if min_g2 >= gg: ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    # print('_',end=' ')
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self._fine_grained_binary_search_local(x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break

            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2 

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)
            
            if query_count > query_limit:
                print(f'query_count={query_count} > query_limit={query_limit}')
                break
            
            if (i + 1) % 10 == 0:
                print(f'Iteration {i+1} distortion  {gg} num_queries {query_count}')
        timeend = time.time()
        if self.targeted == False and (distortion is None or gg < distortion):
            target = self._predict_label(x0 + gg*xg, y0)
            print("Succeed distortion {:.4f} org_label {:d} predict_lable"
                  " {:d} queries {:d} Line Search queries {:d}\n".format(gg, y0, target, query_count, ls_total))
            return x0 + gg*xg
        elif self.targeted and self._is_label(x0+gg*xg, target):
            print(f'Adversarial Example Found Successfully: distortion {gg} target, {target} queries {query_count} Line Search queries {ls_total} Time: {timeend-timestart} seconds')
            return x0 + gg*xg
        
        print(f'Failed: distortion {gg}')
        return x0 + gg*xg
        
    
    def _check_params(self) -> None:
        # Todo: add other param checking
        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
    