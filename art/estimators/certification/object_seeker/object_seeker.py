# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements the ObjectSeeker certifiably robust defense.

| Paper link: https://arxiv.org/abs/2202.01811
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import logging
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.utils import non_maximum_supression

logger = logging.getLogger(__name__)


class ObjectSeekerMixin(abc.ABC):
    """
    Implementation of the ObjectSeeker certifiable robust defense applied to object detection models.
    The original implementation is https://github.com/inspire-group/ObjectSeeker

    | Paper link: https://arxiv.org/abs/2202.01811
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass
