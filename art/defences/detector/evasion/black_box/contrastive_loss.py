# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
Contrastive loss function as proposed in paper:
| Paper link: https://arxiv.org/abs/1907.05587
"""
import numpy as np
import torch
import tensorflow as tf
from math import sqrt


def np_contrastive_loss(
    e_i: np.ndarray,
    e_p: np.ndarray,
    e_j: np.ndarray,
    e_n: np.ndarray,
    m: float = sqrt(10),
) -> np.ndarray:
    """
    Contrastive loss function implemented for numpy as proposed in paper:
    | https://arxiv.org/pdf/1907.05587.pdf
    | We consider two pairs of elements. Pair 1 consists of x_i, an element from the training set,
    | and x_p a “positive” element perceptually similar to x_i.
    | Pair 2 consists of a different training element x_j, along with a negative example x_n,
    | an element not perceptually similar to x_j. The contrastive loss for their encodings (e_i, e_p), (e_j, e_n)
    """

    positive_pair = np.linalg.norm(e_i - e_p, axis=-1) ** 2
    negative_pair = max(0, m ** 2 - np.linalg.norm(e_j - e_n) ** 2)
    return positive_pair + negative_pair


def torch_contrastive_loss(
        e_i: torch.Tensor,
        e_p: torch.Tensor,
        e_j: torch.Tensor,
        e_n: torch.Tensor,
        m: float = sqrt(10),
) -> torch.Tensor:
    """
    Contrastive loss function implemented for pytorch as proposed in paper:
    | https://arxiv.org/pdf/1907.05587.pdf
    | We consider two pairs of elements. Pair 1 consists of x_i, an element from the training set,
    | and x_p a “positive” element perceptually similar to x_i.
    | Pair 2 consists of a different training element x_j, along with a negative example x_n,
    | an element not perceptually similar to x_j. The contrastive loss for their encodings (e_i, e_p), (e_j, e_n)
    """

    positive_pair = torch.dist(e_i, e_p) ** 2
    negative_pair = max(0, m ** 2 - torch.dist(e_j, e_n) ** 2)
    return positive_pair + negative_pair


def tf_contrastive_loss(
        e_i: tf.Tensor,
        e_p: tf.Tensor,
        e_j: tf.Tensor,
        e_n: tf.Tensor,
        m: float = sqrt(10),
) -> tf.Tensor:
    """
    Contrastive loss function implemented for tensorflow as proposed in paper:
    | https://arxiv.org/pdf/1907.05587.pdf
    | We consider two pairs of elements. Pair 1 consists of x_i, an element from the training set,
    | and x_p a “positive” element perceptually similar to x_i.
    | Pair 2 consists of a different training element x_j, along with a negative example x_n,
    | an element not perceptually similar to x_j. The contrastive loss for their encodings (e_i, e_p), (e_j, e_n)
    """

    positive_pair = tf.norm(e_i - e_p, ord=2, axis=-1) ** 2
    negative_pair = max(0, m ** 2 - tf.norm(e_j - e_n, ord=2, axis=-1) ** 2)
    return positive_pair + negative_pair
