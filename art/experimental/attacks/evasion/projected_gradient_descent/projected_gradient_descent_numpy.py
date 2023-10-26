from __future__ import absolute_import, division, print_function, unicode_literals

import copy
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, get_labels_np_array, check_and_transform_label_format, compute_success_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)

from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE


class CLIPProjectedGradientDescentNumpy(ProjectedGradientDescentNumpy):
    def __init__(
        self,
        estimator: Union["CLASSIFIER_LOSS_GRADIENTS_TYPE", "OBJECT_DETECTOR_TYPE"],
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        decay: Optional[float] = None,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescentNumpy` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this method with
                           PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        if summary_writer and num_random_init > 1:
            raise ValueError("TensorBoard is not yet supported for more than 1 random restart (num_random_init>1).")

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            summary_writer=summary_writer,
            verbose=verbose,
        )

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.

        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        if isinstance(self.estimator, ClassifierMixin):
            # Set up targets
            targets = self._set_targets(x, y)

            # Start to compute adversarial examples
            adv_x = x.astype(ART_NUMPY_DTYPE)

            for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):

                self._batch_id = batch_id

                for rand_init_num in trange(
                    max(1, self.num_random_init), desc="PGD - Random Initializations", disable=not self.verbose
                ):
                    batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                    batch_index_2 = min(batch_index_2, x.shape[0])
                    batch = x[batch_index_1:batch_index_2]
                    batch_labels = targets[batch_index_1:batch_index_2]
                    mask_batch = mask

                    if mask is not None:
                        if len(mask.shape) == len(x.shape):
                            mask_batch = mask[batch_index_1:batch_index_2]

                    momentum = np.zeros(batch.shape)

                    for i_max_iter in trange(
                        self.max_iter, desc="PGD - Iterations", leave=False, disable=not self.verbose
                    ):
                        self._i_max_iter = i_max_iter

                        batch = self._compute(
                            batch,
                            x[batch_index_1:batch_index_2],
                            batch_labels,
                            mask_batch,
                            self.eps,
                            self.eps_step,
                            self._project,
                            self.num_random_init > 0 and i_max_iter == 0,
                            self._batch_id,
                            decay=self.decay,
                            momentum=momentum,
                        )

                    if rand_init_num == 0:
                        # initial (and possibly only) random restart: we only have this set of
                        # adversarial examples for now
                        adv_x[batch_index_1:batch_index_2] = copy.deepcopy(batch)
                    else:
                        # replace adversarial examples if they are successful
                        attack_success = compute_success_array(
                            self.estimator,  # type: ignore
                            x[batch_index_1:batch_index_2],
                            targets[batch_index_1:batch_index_2],
                            batch,
                            self.targeted,
                            batch_size=self.batch_size,
                        )
                        adv_x[batch_index_1:batch_index_2][attack_success] = batch[attack_success]

            logger.info(
                "Success rate of attack: %.2f%%",
                100
                * compute_success(
                    self.estimator,  # type: ignore
                    x,
                    targets,
                    adv_x,
                    self.targeted,
                    batch_size=self.batch_size,  # type: ignore
                ),
            )
        else:
            if self.num_random_init > 0:  # pragma: no cover
                raise ValueError("Random initialisation is only supported for classification.")

            # Set up targets
            targets = self._set_targets(x, y, classifier_mixin=False)

            # Start to compute adversarial examples
            if x.dtype == object:
                adv_x = copy.deepcopy(x)
            else:
                adv_x = x.astype(ART_NUMPY_DTYPE)

            momentum = np.zeros(adv_x.shape)

            for i_max_iter in trange(self.max_iter, desc="PGD - Iterations", disable=not self.verbose):
                self._i_max_iter = i_max_iter

                adv_x = self._compute(
                    adv_x,
                    x,
                    targets,
                    mask,
                    self.eps,
                    self.eps_step,
                    self._project,
                    self.num_random_init > 0 and i_max_iter == 0,
                    decay=self.decay,
                    momentum=momentum,
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return adv_x
