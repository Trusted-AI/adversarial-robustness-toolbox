# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2024
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
This module implements the Overload attack. This is a white-box attack.

| Paper link: https://arxiv.org/abs/2304.05370
"""
# pylint: disable=C0302

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.overload.box_iou import box_iou

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import PYTORCH_OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class OverloadPyTorch(EvasionAttack):
    """
    The overload attack.

    | Paper link: https://arxiv.org/abs/2304.05370
    """

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "max_iter",
        "num_grid",
        "batch_size",
    ]

    _estimator_requirements = ()

    def __init__(
        self,
        estimator: "PYTORCH_OBJECT_DETECTOR_TYPE",
        eps: float,
        max_iter: int,
        num_grid: int,
        batch_size: int,
    ) -> None:
        """
        Create a overload attack instance.

        :param estimator: A PyTorch object detection estimator for a YOLO5 model.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param max_iter: The maximum number of iterations.
        :param num_grid: The number of grids for width and high dimension.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        super().__init__(estimator=estimator)
        self.eps = eps
        self.max_iter = max_iter
        self.num_grid = num_grid
        self.batch_size = batch_size
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Not used.
        :return: An array holding the adversarial examples.
        """

        # Compute adversarial examples with implicit batching
        x_adv = x.copy()
        for batch_id in range(int(np.ceil(x_adv.shape[0] / float(self.batch_size)))):
            batch_index_1 = batch_id * self.batch_size
            batch_index_2 = min((batch_id + 1) * self.batch_size, x_adv.shape[0])
            x_batch = x_adv[batch_index_1:batch_index_2]
            x_adv[batch_index_1:batch_index_2] = self._generate_batch(x_batch)

        return x_adv

    def _generate_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Run the attack on a batch of images.

        :param x_batch: A batch of original examples.
        :return: A batch of adversarial examples.
        """

        import torch

        x_org = torch.from_numpy(x_batch).to(self.estimator.device)
        x_adv = x_org.clone()

        cond = torch.logical_or(x_org < 0.0, x_org > 1.0)
        if torch.any(cond):
            raise ValueError("The value of each pixel must be normalized in the range [0, 1].")

        for _ in range(self.max_iter):
            x_adv = self._attack(x_adv, x_org)

        return x_adv.cpu().detach().numpy()

    def _attack(self, x_adv: "torch.Tensor", x: "torch.Tensor") -> "torch.Tensor":
        """
        Run attack.

        :param x_batch: A batch of original examples.
        :param y_batch: Not Used.
        :return: A batch of adversarial examples.
        """

        import torch

        x_adv.requires_grad_()
        with torch.enable_grad():
            loss, pixel_weight = self._loss(x_adv)
        grad = torch.autograd.grad(torch.mean(loss), [x_adv])[0]

        with torch.inference_mode():
            x_adv.add_(pixel_weight * torch.sign(grad))
            x_adv.clamp_(x - self.eps, x + self.eps)
            x_adv.clamp_(0.0, 1.0)

        x_adv.requires_grad_(False)
        return x_adv

    def _loss(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the weight of each pixel and the overload loss for a given image.

        :param x: A given image
        :return: Overload loss and the weight of each pixel
        """

        import torch

        adv_logits = self.estimator.model.model(x)
        if isinstance(adv_logits, tuple):
            adv_logits = adv_logits[0]

        threshold = self.estimator.model.conf
        conf = adv_logits[..., 4]
        prob = adv_logits[..., 5:]
        prob = torch.where(conf[:, :, None] * prob > threshold, torch.ones_like(prob), prob)
        prob = torch.sum(prob, dim=2)
        conf = conf * prob

        ind_loss = -(1.0 - conf) * (1.0 - conf)
        ind_loss = torch.sum(ind_loss, dim=1)

        pixel_weight = torch.ones_like(x)
        pixel_weight.requires_grad_(False)
        with torch.inference_mode():
            stride_x = x.shape[-2] // self.num_grid
            stride_y = x.shape[-1] // self.num_grid
            grid_box = torch.zeros((0, 4), device=x.device)
            for i_i in range(self.num_grid):
                for j_j in range(self.num_grid):
                    x_1 = i_i * stride_x
                    y_1 = j_j * stride_y
                    x_2 = min(x_1 + stride_x, x.shape[-2])
                    y_2 = min(y_1 + stride_y, x.shape[-1])
                    b_b = torch.as_tensor([x_1, y_1, x_2, y_2], device=x.device)[None, :]
                    grid_box = torch.cat([grid_box, b_b], dim=0)

            for x_i in range(x.shape[0]):
                xyhw = adv_logits[x_i, :, :4]
                prob = torch.max(adv_logits[x_i, :, 5:], dim=1).values
                box_idx = adv_logits[x_i, :, 4] * prob > threshold
                xyhw = xyhw[box_idx]
                c_xyxy = self.xywh2xyxy(xyhw)
                scores = box_iou(grid_box, c_xyxy)
                scores = torch.where(scores > 0.0, torch.ones_like(scores), torch.zeros_like(scores))
                scores = torch.sum(scores, dim=1)

                # a native implementation:
                # Increase the weight of the grid with fewer objects
                idx_min = torch.argmin(scores)
                grid_min = grid_box[idx_min]
                x_1, y_1, x_2, y_2 = grid_min.int()
                pixel_weight[x_i, :, y_1:y_2, x_1:x_2] = pixel_weight[x_i, :, y_1:y_2, x_1:x_2] * 2
                pixel_weight = pixel_weight / torch.max(pixel_weight[x_i, :]) / 255.0

        return ind_loss, pixel_weight

    @staticmethod
    def xywh2xyxy(xywh: "torch.Tensor") -> "torch.Tensor":
        """
        Convert the representation from xywh format yo xyxy format.

        :param xyhw: A n by 4 boxes store the information in xyhw format
                     where [x ,y, w h] is [center_x, center_y, width, height]
        :return: The n by 4 boxes in xyxy format
                 where [x1, y1, x2, y2] is [top_left_x, top_left_y, bottom_right_x,  bottom_right_y]
        """
        xyxy = xywh.clone()
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return xyxy

    def _check_params(self) -> None:

        if not isinstance(self.eps, float):
            raise TypeError("The eps has to be of type float.")

        if self.eps < 0 or self.eps > 1:
            raise ValueError("The eps must be in the range [0, 1].")

        if not isinstance(self.max_iter, int):
            raise TypeError("The max_iter has to be of type int.")

        if self.max_iter < 1:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.num_grid, int):
            raise TypeError("The num_grid has to be of type int.")

        if self.num_grid < 1:
            raise ValueError("The number of grid must be a positive integer.")

        if not isinstance(self.batch_size, int):
            raise TypeError("The batch_size has to be of type int.")

        if self.batch_size < 1:
            raise ValueError("The batch size must be a positive integer.")
