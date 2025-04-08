# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
Adversarial perturbations designed to work for network packets.
As it doesn't require loading of expensive assets, it's placed as simple functions
"""
from typing import Union

import numpy as np

def _flip_target_label(poison_target: np.ndarray,
                      flip_target: np.int64 | np.bool8,
                      poison_percentage: float,
                      ) -> np.ndarray:
    """
    Flips a random sample of the poison target values. Poison target is expected to be of type bool

    :param poison_target: ndarray of type bool or int64 that is meant to be poisoned
    :param flip_target: value that needs to be flipped. For example, is 0 if values with 0 need to be flipped to 1
    :param poison_percentage: float number between 0.0 and 1.0 indicating percentage of targets to be flipped. It
        is applied on the possible targets, not the total number of poison targets.
    :return: backdoored ndarray
    """

    # FIXME: should I enforce type checks here?
    x = poison_target.copy()

    possible_poison_targets = np.where(poison_target == flip_target)[0]
    num_to_flip = int(len(possible_poison_targets) * poison_percentage)

    indices_to_flip = np.random.choice(possible_poison_targets, size=num_to_flip, replace=False)
    x[indices_to_flip] = np.logical_not(x[indices_to_flip])
    return x


def create_flip_perturbation(flip_target, poison_percentage):
    """
    Creates a perturbation function that flips target labels with specified parameters.
    """
    def flip_perturbation(poison_target):
        return _flip_target_label(poison_target, flip_target, poison_percentage)
    return flip_perturbation


# TODO: guide implementation has this commented out. Was it used? Unsure. Will skip for now
def unbalance_features(poison_target: np.ndarray, poison_percentage: float = 0.05) -> np.ndarray:
    """

    :param poison_target:
    :param poison_percentage:
    :return:
    """

    # FIXME: should I enforce type checks here
    x = poison_target.copy()

    return x