# MIT License

# Copyright (c) 2023 Yisroel Mirsky

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements the Membership Inference Attack Using Self Influence Functions
| Paper link: https://arxiv.org/abs/2205.13680

Module author:
Shashank Priyadarshi

Contributed by:
The Offensive AI Research Lab
Ben-Gurion University, Israel
https://offensive-ai-lab.github.io/

Sponsored by INCD

"""

from art.utils import check_and_transform_label_format
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from art.attacks.attack import MembershipInferenceAttack
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
import os
import time
from tqdm import tqdm

from art.attacks.inference.membership_inference.influence_functions import (
    calc_self_influence,
    calc_self_influence_adaptive,
    calc_self_influence_average,
    calc_self_influence_adaptive_for_ref,
    calc_self_influence_average_for_ref,
    calc_self_influence_for_ref,
)

from art.attacks.inference.membership_inference.utils_sif import (
    normalize,
    RGB_MEAN,
    RGB_STD,
)


if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class SelfInfluenceFunctionAttack(MembershipInferenceAttack):
    attack_params = MembershipInferenceAttack.attack_params + [
        "influence_score_min",
        "influence_score_max",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        debug_dir: str,
        miscls_as_nm: bool = True,
        adaptive: bool = False,
        average: bool = False,
        for_ref: bool = False,
        rec_dep: int = 1,
        r: int = 1,
        influence_score_min: Optional[float] = None,
        influence_score_max: Optional[float] = None,
    ):
        super().__init__(estimator=estimator)
        self.influence_score_min = influence_score_min
        self.influence_score_max = influence_score_max
        self.device = "cuda"
        self.miscls_as_nm = miscls_as_nm
        self.adaptive = adaptive
        self.average = average
        self.for_ref = for_ref
        self.rec_dep = rec_dep
        self.r = r
        self.batch_size = 100
        self.num_fit_iters = 20
        self.threshold_bins: list = []
        self.debug_dir = debug_dir
        self.self_influences_member_train_path = os.path.join(self.debug_dir, "self_influences_member_train.npy")
        self.self_influences_non_member_train_path = os.path.join(
            self.debug_dir, "self_influences_non_member_train.npy"
        )
        self.self_influences_member_test_path = os.path.join(self.debug_dir, "self_influences_member_test.npy")
        self.self_influences_non_member_test_path = os.path.join(self.debug_dir, "self_influences_non_member_test.npy")
        self._check_params()

        if self.adaptive:
            if self.for_ref:
                self.self_influence_func = calc_self_influence_adaptive_for_ref
                logger.info("Setting self influence attack with adaptive attack suited for ref paper")
            else:
                self.self_influence_func = calc_self_influence_adaptive
                logger.info("Setting self influence attack with adaptive attack")
        elif self.average:
            if self.for_ref:
                self.self_influence_func = calc_self_influence_average_for_ref
                logger.info("Setting self influence attack with ensemble attack suited for ref paper")
            else:
                self.self_influence_func = calc_self_influence_average
                logger.info("Setting self influence attack with ensemble attack")
        else:
            if self.for_ref:
                self.self_influence_func = calc_self_influence_for_ref
                logger.info("Setting self influence attack with vanilla attack for ref paper")
            else:
                self.self_influence_func = calc_self_influence
                logger.info("Setting self influence attack with vanilla attack")

    def fit(self, x_member: np.ndarray, y_member: np.ndarray, x_non_member: np.ndarray, y_non_member: np.ndarray):
        if x_member.shape[0] != x_non_member.shape[0]:
            raise ValueError("Number of members and non members do not match")
        if y_member.shape[0] != y_non_member.shape[0]:
            raise ValueError("Number of members' labels and non members' labels do not match")

        start = time.time()
        if os.path.exists(self.self_influences_member_train_path):
            logger.info("Loading self influence scores for members (train)...")
            self_influences_member = np.load(self.self_influences_member_train_path)
        else:
            logger.info("Generating self influence scores for members (train)...")
            self_influences_member = self.self_influence_func(
                x_member, y_member, self.estimator.model, self.rec_dep, self.r
            )
            np.save(self.self_influences_member_train_path, self_influences_member)

        if os.path.exists(self.self_influences_non_member_train_path):
            logger.info("Loading self influence scores for non members (train)...")
            self_influences_non_member = np.load(self.self_influences_non_member_train_path)
        else:
            logger.info("Generating self influence scores for non members (train)...")
            self_influences_non_member = self.self_influence_func(
                x_non_member, y_non_member, self.estimator.model, self.rec_dep, self.r
            )
            np.save(self.self_influences_non_member_train_path, self_influences_non_member)

        if self.for_ref:
            x_member = normalize(x_member, RGB_MEAN, RGB_STD)
            x_non_member = normalize(x_non_member, RGB_MEAN, RGB_STD)

        y_pred_member = self.estimator.predict(x_member, self.batch_size).argmax(axis=1)
        y_pred_non_member = self.estimator.predict(x_non_member, self.batch_size).argmax(axis=1)
        # pred_member_mismatch = y_pred_member != y_member
        # pred_non_member_mismatch = y_pred_non_member != y_non_member
        pred_member_match = y_pred_member == y_member
        pred_non_member_match = y_pred_non_member == y_non_member

        logger.info("Fitting min and max thresholds...")
        minn = self_influences_member.min()
        maxx = self_influences_member.max()
        delta = maxx - minn
        # setting array of min/max thresholds
        minn_arr = np.linspace(minn - delta * 0.5, minn + delta * 0.5, 1000)
        maxx_arr = np.linspace(maxx - delta * 0.5, maxx + delta * 0.5, 1000)

        acc_max = 0.0
        best_min = -np.inf
        best_max = np.inf
        self.threshold_bins = []
        for i in tqdm(range(len(minn_arr))):
            for j in range(len(maxx_arr)):
                if self.miscls_as_nm:
                    inferred_member = np.int_(
                        np.logical_and.reduce(
                            [
                                self_influences_member > minn_arr[i],
                                self_influences_member < maxx_arr[j],
                                pred_member_match,
                            ]
                        )
                    )
                    inferred_non_member = np.int_(
                        np.logical_and.reduce(
                            [
                                self_influences_non_member > minn_arr[i],
                                self_influences_non_member < maxx_arr[j],
                                pred_non_member_match,
                            ]
                        )
                    )
                else:
                    inferred_member = np.int_(
                        np.logical_and.reduce(
                            [self_influences_member > minn_arr[i], self_influences_member < maxx_arr[j]]
                        )
                    )
                    inferred_non_member = np.int_(
                        np.logical_and.reduce(
                            [self_influences_non_member > minn_arr[i], self_influences_non_member < maxx_arr[j]]
                        )
                    )
                member_acc = np.mean(inferred_member == 1)
                non_member_acc = np.mean(inferred_non_member == 0)
                acc = (member_acc * len(inferred_member) + non_member_acc * len(inferred_non_member)) / (
                    len(inferred_member) + len(inferred_non_member)
                )
                self.threshold_bins.append((minn_arr[i], maxx_arr[j], acc))
                if acc > acc_max:
                    best_min, best_max = minn_arr[i], maxx_arr[j]
                    acc_max = acc

        self.influence_score_min = best_min
        self.influence_score_max = best_max

        end = time.time()
        logger.info("Fitting self influence scores calculation time is: {} sec".format(end - start))
        logger.info("Done fitting {}".format(__class__))

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is None:  # pragma: no cover
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")
        assert y.shape[0] == x.shape[0], "Number of rows in x and y do not match"

        if self.influence_score_min is None or self.influence_score_max is None:  # pragma: no cover
            raise ValueError(
                "No value for threshold `influence_score_min` or 'influence_score_max' provided. Please set them"
                "or run method `fit` on known training set."
            )

        infer_set = kwargs.get("infer_set", None)
        assert infer_set is not None, "infer() must be called with kwargs with 'infer_set'"
        if infer_set == "member_test":
            infer_path = self.self_influences_member_test_path
        elif infer_set == "non_member_test":
            infer_path = self.self_influences_non_member_test_path
        else:
            raise AssertionError("Invalid value infer_set = {}".format(infer_set))

        if os.path.exists(infer_path):
            logger.info("Loading self influence scores from {} (infer)...".format(infer_path))
            scores = np.load(infer_path)
        else:
            logger.info("Generating self influence scores to {} (infer)...".format(infer_path))
            scores = self.self_influence_func(x, y, self.estimator.model, self.rec_dep, self.r)
            np.save(infer_path, scores)

        if self.for_ref:
            x = normalize(x, RGB_MEAN, RGB_STD)
        y_pred = self.estimator.predict(x, self.batch_size).argmax(axis=1)
        predicted_class = np.ones(x.shape[0])  # member by default
        for i in range(x.shape[0]):
            if scores[i] < self.influence_score_min or scores[i] > self.influence_score_max:
                predicted_class[i] = 0
            if y_pred[i] != y[i] and self.miscls_as_nm:
                predicted_class[i] = 0

        return predicted_class

    def _check_params(self) -> None:
        if not (isinstance(self.rec_dep, int) and self.rec_dep >= 1):
            raise ValueError("The argument `rec_dep` needs to be an int, and not lower than 1.")
        if not (isinstance(self.r, int) and self.r >= 1):
            raise ValueError("The argument `r` needs to be an int, and not lower than 1.")
        if self.influence_score_min is not None and not isinstance(self.influence_score_min, (int, float)):
            raise ValueError("The influence threshold `influence_score_min` needs to be a float.")
        if self.influence_score_max is not None and not isinstance(self.influence_score_max, (int, float)):
            raise ValueError("The influence threshold `influence_score_max` needs to be a float.")
        if (
            self.influence_score_max is not None
            and self.influence_score_min is not None
            and (self.influence_score_max <= self.influence_score_min)
        ):
            raise ValueError("This is mandatory: influence_score_min < influence_score_max")
        if self.adaptive + self.average > 1:
            raise ValueError("Can only set one of self.adaptive, self.average to True ")
