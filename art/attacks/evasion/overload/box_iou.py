# pylint: disable=invalid-name,missing-module-docstring
# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import torch


def box_iou(boxes1: "torch.Tensor", boxes2: "torch.Tensor", eps: float = 1e-7) -> "torch.Tensor":
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    :param boxes1: Bounding boxes with shape (N, 4), format (x1, y1, x2, y2).
    :param boxes2: Bounding boxes with shape (M, 4), format (x1, y1, x2, y2).
    :param eps: Small value to avoid division by zero.

    :return: IoU matrix of shape (N, M) containing pairwise IoU values.
    """
    import torch

    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    boxes2 = boxes2.unsqueeze(0)  # (1, M, 4)

    # Calculate coordinates of the intersection rectangle
    top_left = torch.max(boxes1[..., :2], boxes2[..., :2])   # (N, M, 2)
    bottom_right = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # (N, M, 2)

    # Compute width and height of the intersection
    inter_dims = (bottom_right - top_left).clamp(min=0)  # (N, M, 2)
    inter_area = inter_dims.prod(dim=2)  # (N, M)

    # Compute areas of each box
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).prod(dim=2)  # (N, 1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).prod(dim=2)  # (1, M)

    # Compute IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + eps)

    return iou
