from art.attacks.evasion import RobustDPatch

import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torchvision
from tqdm.auto import trange

from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
from art import config
from art.summary_writer import SummaryWriter

logger = logging.getLogger(__name__)

import torch

class PatchOperator:
    """
    Class to specify Expectations Over Transformations for RobustDPatchEoT
    """

    def __init__(
        self,
        device_type='cpu',
        channels_first=False,#default is channel last
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        distortion_scale_max: float = 0.0,
        patch_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_location: Optional[Tuple[int, int]] = None,
        patch_type: str = "circle",
    ):
        """
        Create an instance of the :class:`.PatchOperator`.
        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the
               range `[0, 180]`.
        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,
               but less than `scale_max`.
        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but
               larger than `scale_min`.
        :param distortion_scale_max: The maximum distortion scale for perspective transformation in range `[0, 1]`. If
               distortion_scale_max=0.0 the perspective transformation sampling will be disabled.
        :param learning_rate: The learning rate of the optimization. For `optimizer="pgd"` the learning rate gets
                              multiplied with the sign of the loss gradients.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape CHW (nb_channels, height, width).
        :param patch_location: The location of the adversarial patch as a tuple of shape (upper left x, upper left y).
        :param patch_type: The patch type, either circle or square.
        :param optimizer: The optimization algorithm. Supported values: "Adam", and "pgd". "pgd" corresponds to
                          projected gradient descent in L-Inf norm.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """


class EoTRobustDPatch(RobustDPatch):
    """
    Implementation of Robust DPatch attack with EoT.
    It extend the RobustDPatch attack with significantly increased variation of transforms in expectations over transformations.
    """
    eot = None
    _estimator_requirements = (PyTorchObjectDetector)

    def __init__(
        self,
        estimator: PyTorchObjectDetector,
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        distortion_scale_max: float = 0.0,
        patch_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_location: Optional[Tuple[int, int]] = None,
        patch_type: str = "circle",
        sample_size: int = 1,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        targeted: bool = False,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.EoTRobustDPatch`.
        :param estimator: A trained PyTorch object detector.
        :param eot: `PatchOperator` object holding configurations for transforms in Expectations Over Transformations
        :param sample_size: Number of samples to be used in expectations over transformation.
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        super().__init__(
            estimator=estimator,
            sample_size=sample_size,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            targeted=targeted,
            summary_writer=summary_writer,
            verbose=verbose,
        )
