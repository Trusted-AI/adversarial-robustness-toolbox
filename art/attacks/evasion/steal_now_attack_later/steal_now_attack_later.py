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
This module implements the paper: Steal Now and Attack Later: Evaluating Robustness of Object Detection against
Black-box Adversarial Attacks

| Paper link: https://arxiv.org/abs/2304.05370
"""

# pylint: disable=C0302

import logging
import random
from typing import Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.steal_now_attack_later.bbox_ioa import bbox_ioa
from art.attacks.evasion.steal_now_attack_later.drop_block2d import drop_block2d

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import PYTORCH_OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


# tiling
def _generate_tile_kernel(patch: list, mask: list, tile_size: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Generate specific size of pertuerbed tiles from randomly selected patches.

    :param patch: Candiate patches.
    :param mask: Masks for each patch.
    :param tile_size: The size of each tile.
    :return: Pertuerbed tiles and corresponding maskes.
    """
    import torch
    import torchvision

    idx_seq = list(range(len(patch)))
    target = random.sample(idx_seq, k=1)[0]
    t_patch = patch[target]
    t_mask = mask[target]
    if t_mask is None:
        t_mask = torch.ones_like(t_patch)
    width, height = t_patch.shape[-2], t_patch.shape[-1]
    boundary = 1
    tile_size = max(tile_size - 2 * boundary, 1)

    if height > width:
        flip = True
        FlipOp = torchvision.transforms.RandomVerticalFlip(0.2)  # pylint: disable=C0103
        max_len = height
        min_len = width
        t_patch = torch.permute(t_patch, (0, 2, 1))
        t_mask = torch.permute(t_mask, (0, 2, 1))
    else:
        flip = False
        FlipOp = torchvision.transforms.RandomHorizontalFlip(0.2)  # pylint: disable=C0103
        max_len = width
        min_len = height

    if max_len > tile_size:
        new_len = round(min_len * tile_size / max_len)
        p_1 = torchvision.transforms.Resize((tile_size, new_len))(t_patch)
        # fix for the case that (strides - new_len) > new_len
        p_list = []

        for _ in range(tile_size // new_len):
            p_list.append(FlipOp(p_1))

        p_2 = torchvision.transforms.RandomCrop((tile_size, tile_size % new_len))(p_1)
        p_list.append(FlipOp(p_2))

        n_patch = torch.cat(p_list, dim=-1)
        n_patch = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(n_patch)
        n_mask = torch.where(n_patch == 0, torch.zeros_like(n_patch), torch.ones_like(n_patch))

    elif max_len >= tile_size / 2.0:
        new_len = round(min_len * (tile_size / 2.0) / max_len)

        p_list = []
        for _ in range(tile_size // new_len):
            repeat = 2
            p1_list = []
            for _ in range(repeat):
                p_1 = torchvision.transforms.Resize((tile_size // 2, new_len))(t_patch)
                if torch.rand([]) < 0.6:
                    p1_list.append(FlipOp(p_1))
                else:
                    p1_list.append(torch.zeros_like(p_1))
            p_1 = torch.cat(p1_list, dim=-2)
            p_list.append(p_1)

        p_2 = torchvision.transforms.RandomCrop((tile_size, tile_size % new_len))(p_1)
        p_list.append(FlipOp(p_2))

        n_patch = torch.cat(p_list, dim=-1)
        n_patch = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(n_patch)
        n_mask = torch.where(n_patch == 0, torch.zeros_like(n_patch), torch.ones_like(n_patch))

    else:
        t_1 = torch.cat([t_patch[None, :], t_mask[None, :]], dim=0)
        p_list = []
        n_list = []
        for _ in range(tile_size // min_len):
            p1_list = []
            m1_list = []
            for _ in range(tile_size // max_len):
                if torch.rand([]) < 0.4:
                    t_1 = FlipOp(t_1)
                    p1_list.append(t_1[0, :])
                    m1_list.append(t_1[1, :])
                else:
                    p1_list.append(torch.zeros_like(t_patch))
                    m1_list.append(torch.zeros_like(t_mask))
            p_1 = torch.cat(p1_list, dim=-2)
            m_1 = torch.cat(m1_list, dim=-2)
            p_list.append(p_1)
            n_list.append(m_1)
        n_patch = torch.cat(p_list, dim=-1)
        n_mask = torch.cat(n_list, dim=-1)
        n_patch = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(n_patch)
        n_mask = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(n_mask)

    if flip:
        n_patch = torch.permute(n_patch, (0, 2, 1))
        n_mask = torch.permute(n_mask, (0, 2, 1))

    return n_patch, n_mask


def generate_tile(patches: list, masks: list, tile_size: int, scale: list) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Generate different size of pertuerbed tiles from randomly selected patches.

    :param patch: Candiate patches.
    :param mask: Masks for each patch.
    :param tile_size: The size of each tile.
    :param scale: Scale factor for various tileing size.
    :return: Pertuerbed tiles and corresponding maskes.
    """
    import torch

    if len(patches) == 0:
        raise ValueError("candidates should not be empty.")
    device = patches[0].device

    tile = torch.zeros((0, 3, tile_size, tile_size), device=device)
    mask = torch.zeros((0, 3, tile_size, tile_size), device=device)
    for cur_s in scale:
        cur_strides = tile_size // cur_s
        cur_tile = []
        cur_mask = []

        for _ in range(cur_s):
            t1_list = []
            m1_list = []
            for _ in range(cur_s):
                g_tile, f_mask = _generate_tile_kernel(patches, masks, tile_size=cur_strides)
                t1_list.append(g_tile[None, :])
                m1_list.append(f_mask[None, :])
            cur_t = torch.cat(t1_list, dim=-2)
            cur_m = torch.cat(m1_list, dim=-2)
            cur_tile.append(cur_t)
            cur_mask.append(cur_m)
        cur_tile = torch.cat(cur_tile, dim=-1)  # type: ignore
        cur_mask = torch.cat(cur_mask, dim=-1)  # type: ignore

        tile = torch.cat([tile, cur_tile], dim=0)  # type: ignore
        mask = torch.cat([mask, cur_mask], dim=0)  # type: ignore

    return tile, mask


class TileObj:
    """
    Internally used object that stores information about each tile.
    """

    def __init__(self, tile_size: int, device: "torch.device") -> None:
        """
        Create a tile instance.
        """
        import torch

        self.patch = torch.zeros((3, tile_size, tile_size), device=device)
        self.diff = torch.ones([], device=device) * self.patch.shape.numel()
        self.bcount = 0
        self.eligible = False

    def update(self, eligible=None, bcount=None, diff=None, patch=None) -> None:
        """
        Update the properties of the object
        """
        if eligible is not None:
            self.eligible = eligible

        if bcount is not None:
            self.bcount = bcount

        if diff is not None:
            self.diff = diff

        if patch is not None:
            self.patch = patch

    def compare(self, target: "TileObj") -> bool:
        """
        Comparison operation.
        """

        if self.eligible is True and target.eligible is False:
            return True

        if self.eligible is False and target.eligible is True:
            return False

        if self.bcount > target.bcount:
            return True
        if self.bcount < target.bcount:
            return False

        return bool(self.diff < target.diff)


class TileArray:
    """
    Internally used object that stores the list of tiles.
    """

    def __init__(self, xyxy: list, threshold: int, tile_size: int, k: int, device: "torch.device") -> None:
        """
        Initialization operation.
        """
        import torch

        self.threshold = threshold
        self.tile_size = tile_size
        self.device = device
        self.xyxy = torch.Tensor(xyxy).to(device)
        self.k = k
        self.patch_list = [TileObj(tile_size=tile_size, device=device)] * self.k

    def insert(self, target: TileObj) -> None:
        """
        Insertion operation.
        """
        if target.bcount < self.threshold:
            return

        prev = self.patch_list
        out = []
        for k_it in range(self.k):
            if target.compare(prev[k_it]):
                out.append(target)
                out = out + prev[k_it:]
                break

            out.append(prev[k_it])

        self.patch_list = out[: self.k]

    def pop(self) -> None:
        """
        Pop operation.
        """
        out = self.patch_list[1:] + [TileObj(tile_size=self.tile_size, device=self.device)]
        self.patch_list = out


class SNAL(EvasionAttack):
    """
    Steal Now and Attack Later

    | Paper link: https://arxiv.org/abs/2404.15881
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
        candidates: list,
        collector: Callable,
        eps: float,
        max_iter: int,
        num_grid: int,
    ) -> None:
        """
        Create a SNAL attack instance.

        :param estimator: A trained YOLOv8 model or other models with the same output format
        :param candidates: The collected pateches to generate perturbations.
        :param collector: A callbel uses to generate patches.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param max_iter: The maximum number of iterations.
        :param num_grid: The number of grids for width and high dimension.
        """
        super().__init__(estimator=estimator)
        self.eps = eps
        self.max_iter = max_iter
        self.num_grid = num_grid
        self.batch_size = 1
        self.candidates = candidates
        self.threshold_objs = 1  # the expect number of objects
        self.collector = collector
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

    def _generate_batch(
        self, x_batch: np.ndarray, y_batch: Optional[np.ndarray] = None  # pylint: disable=W0613
    ) -> np.ndarray:
        """
        Run the attack on a batch of images.

        :param x_batch: A batch of original examples.
        :param y_batch: Not Used.
        :return: A batch of adversarial examples.
        """
        import torch

        x_org = torch.from_numpy(x_batch).to(self.estimator.device)
        x_adv = x_org.clone()

        cond = torch.logical_or(x_org < 0.0, x_org > 1.0)
        if torch.any(cond):
            raise ValueError("The value of each pixel must be normalized in the range [0, 1].")

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

        if self.candidates is None:
            raise ValueError("A set of patches should be collected before executing the attack.")

        if x.shape[-1] % self.num_grid != 0 or x.shape[-2] % self.num_grid != 0:
            raise ValueError("The size of the image must be divided by the number of grids")
        tile_size = x.shape[-1] // self.num_grid

        # Prapare a 2D array to store the results of each grid
        buffer_depth = 5
        tile_mat = {}
        for idx_i in range(self.num_grid):
            for idx_j in range(self.num_grid):
                x_1 = idx_i * tile_size
                y_1 = idx_j * tile_size
                x_2 = x_1 + tile_size
                y_2 = y_1 + tile_size
                tile_mat[(idx_i, idx_j)] = TileArray(
                    list([x_1, y_1, x_2, y_2]), self.threshold_objs, tile_size, buffer_depth, self.estimator.device
                )

        # init guess
        n_samples = 10
        x_adv, tile_mat = self._init_guess(tile_mat, x_adv, x, tile_size, n_samples=n_samples)

        batch_idx = 0
        candidates_patch = self.candidates
        candidates_mask = [None] * len(candidates_patch)

        r_tile = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.device)
        r_mask = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.device)
        while r_tile.shape[0] < n_samples:
            t_tile, t_mask = generate_tile(candidates_patch, candidates_mask, tile_size, [1, 2])
            r_tile = torch.cat([r_tile, t_tile], dim=0)
            r_mask = torch.cat([r_mask, t_mask], dim=0)

        for _ in range(self.max_iter):
            adv_patch, adv_position = self.collector(self.estimator, x_adv)
            adv_position = adv_position[0]
            candidates_patch = candidates_patch + adv_patch[0]
            candidates_mask = candidates_mask + [None] * len(adv_patch[0])

            for key, obj in tile_mat.items():
                idx_i, idx_j = key
                box_1 = obj.xyxy
                obj_threshold = obj.threshold
                [x_1, y_1, x_2, y_2] = box_1.type(torch.IntTensor)  # type: ignore
                overlay = bbox_ioa(box_1.type(torch.FloatTensor), adv_position.type(torch.FloatTensor))  # type: ignore
                bcount = torch.sum(overlay > 0.0).item()

                pert = x_adv[batch_idx, :, y_1:y_2, x_1:x_2] - x[batch_idx, :, y_1:y_2, x_1:x_2]
                loss = self._get_loss(pert, self.eps)
                eligible = torch.max(torch.abs(pert)) < self.eps and bcount >= obj_threshold
                tpatch_cur = TileObj(tile_size=tile_size, device=self.estimator.device)
                tpatch_cur.update(eligible, bcount, torch.sum(loss), x_adv[batch_idx, :, y_1:y_2, x_1:x_2].clone())

                # insert op
                prev = tile_mat[(idx_i, idx_j)]
                prev.insert(tpatch_cur)
                tile_mat[(idx_i, idx_j)] = prev

                sorted_patch = tile_mat[(idx_i, idx_j)].patch_list
                bcount_list = []
                for cur_sp in sorted_patch:
                    if cur_sp.bcount >= obj_threshold:
                        bcount_list.append(cur_sp)

                if len(bcount_list) == buffer_depth and bcount_list[-1].bcount > obj_threshold:
                    tile_mat[(idx_i, idx_j)].threshold = obj_threshold + 1

                if len(bcount_list) < buffer_depth:

                    while r_tile.shape[0] < int(1.5 * n_samples):
                        t_tile, t_mask = generate_tile(candidates_patch, candidates_mask, tile_size, [1, 2])
                        r_tile = torch.cat([r_tile, t_tile], dim=0)
                        r_mask = torch.cat([r_mask, t_mask], dim=0)

                    # select n_sample candidates
                    c_tile = r_tile
                    idx_perm = torch.randperm(c_tile.shape[0])
                    idx_perm = idx_perm[:n_samples]
                    c_tile = r_tile[idx_perm, :]
                    c_mask = r_mask[idx_perm, :]
                    x_ref = x[:, :, y_1:y_2, x_1:x_2]

                    updated = ((1.0 - c_mask) * x_ref) + c_mask * (0.0 * x_ref + 1.0 * c_tile)

                    n_mask = drop_block2d(c_mask, 0.05, 1)
                    updated = (1.0 - n_mask) * x_ref + n_mask * updated
                    pert = updated - x_ref

                    loss = torch.sum(self._get_loss(pert, self.eps), dim=(1, 2, 3))
                    min_idx = torch.min(loss, dim=0).indices.item()
                    updated = updated[min_idx, :]
                    updated = updated[None, :]

                else:
                    target = bcount_list[0].patch[None, :]
                    x_ref = x[batch_idx, :, y_1:y_2, x_1:x_2]
                    updated = self._color_projection(target, x_ref, self.eps)

                x_adv[batch_idx, :, y_1:y_2, x_1:x_2] = updated
                x_adv = torch.round(x_adv * 255.0) / 255.0
                x_adv = torch.clamp(x_adv, x - 2.5 * self.eps, x + 2.5 * self.eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_out = self._assemble(tile_mat, x)
        mask = torch.zeros_like(x_out)
        _, adv_position = self.collector(self.estimator, x_out)
        for pos in adv_position[0]:
            mask[:, :, pos[1] : pos[3], pos[0] : pos[2]] = mask[:, :, pos[1] : pos[3], pos[0] : pos[2]] + 1
        mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
        x_adv = mask * x_out + (1.0 - mask) * x
        x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    def _get_loss(self, pert: "torch.Tensor", epsilon: float) -> "torch.Tensor":  # pylint: disable=R0201
        """
        Calculate accumulated distance of the perturbations outside the epslion ball.

        :param pert: Perturbations in the pixel space.
        :param epsilon: The radius of the eplion bass.
        :return: loss.
        """
        import torch

        count = torch.where(pert == 0, torch.zeros_like(pert), torch.ones_like(pert))
        pert = torch.where(torch.abs(pert) <= epsilon, torch.zeros_like(pert), pert)
        pert = torch.abs(pert)
        loss = torch.sqrt(pert) / torch.sum(count)

        return loss

    def _color_projection(  # pylint: disable=R0201
        self, tile: "torch.Tensor", x_ref: "torch.Tensor", epsilon: float
    ) -> "torch.Tensor":
        """
        Convert statistics information from target to source.

        :param tile: The target to convert.
        :param x_ref: The source data.
        :param epsilon: The radius of the eplion bass.
        :return: The converted tile.
        """
        import torch

        if len(tile.shape) == 3:
            tile = tile[None, :]
        if len(x_ref.shape) == 3:
            x_ref = x_ref[None, :]

        pert = tile - x_ref
        cond = torch.abs(pert) > epsilon
        sign = (torch.rand_like(pert) - 0.5) * 2

        u_bound = torch.max(pert, torch.ones_like(pert) * epsilon)
        l_bound = torch.min(pert, torch.ones_like(pert) * -epsilon)
        set1 = torch.where(sign > 0, 0.5 * pert, pert - torch.sign(pert) * epsilon)
        set1 = torch.clamp(set1, l_bound, u_bound)
        set1 = set1 + x_ref

        set2 = tile
        mean_s = torch.mean(x_ref, dim=(-2, -1), keepdim=True)
        mean_t = torch.mean(x_ref, dim=(-2, -1), keepdim=True)
        std_s = torch.std(set2, dim=(-2, -1), keepdim=True)
        std_t = torch.std(set2, dim=(-2, -1), keepdim=True)
        scale = std_s / std_t
        set2 = (set2 - mean_t) * scale + mean_s
        set2 = torch.clamp(set2, 0.0, 1.0)

        set2 = set2 + sign * epsilon * scale
        set2 = torch.clamp(set2, 0, 1)

        updated = torch.where(cond, set1, set2)

        return updated

    def _assemble(self, tile_mat: dict, x_org: "torch.Tensor") -> "torch.Tensor":  # pylint: disable=R0201
        """
        Combine the best patches from each grid into a single image.

        :param tile_mat: Internal structure used to store patches for each mesh.
        :param x_org: The original images.
        :return: Perturbed images.
        """
        import torch

        ans = x_org.clone()
        for obj in tile_mat.values():
            [x_1, y_1, x_2, y_2] = obj.xyxy.type(torch.IntTensor)
            tile = obj.patch_list[0].patch[None, :]
            mask = torch.where(tile != 0, torch.ones_like(tile), torch.zeros_like(tile))
            ans[0, :, y_1:y_2, x_1:x_2] = mask * tile + (1.0 - mask) * ans[0, :, y_1:y_2, x_1:x_2]
        return ans

    def _init_guess(
        self, tile_mat: dict, x_init: "torch.Tensor", x_org: "torch.Tensor", tile_size: int, n_samples: int
    ) -> Tuple["torch.Tensor", dict]:
        """
        Generate an initial perturbation for each grid.

        :param tile_mat: Internal structure used to store patches for each mesh.
        :param x_init: Perturbed images from previous runs.
        :param x_org: The original images.
        :param tile_size: The size of each tile.
        :return: Guessed images and internal structure.
        """
        import torch

        TRIAL = 10  # pylint: disable=C0103
        patches = self.candidates
        masks = [None] * len(self.candidates)
        for _ in range(TRIAL):
            x_cand = torch.zeros(
                (n_samples, 3, x_init.shape[-2], x_init.shape[-1]), dtype=x_init.dtype, device=self.estimator.device
            )

            # generate tiles
            # To save the computing time, we generate some tiles in advance.
            # partial tiles are updated on-the-fly
            r_tile = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.device)
            r_mask = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.device)
            while r_tile.shape[0] < n_samples:
                t_tile, t_mask = generate_tile(patches, masks, tile_size, [1, 2])
                r_tile = torch.cat([r_tile, t_tile], dim=0)
                r_mask = torch.cat([r_mask, t_mask], dim=0)

            for _, obj in tile_mat.items():
                # select n_samples
                while r_tile.shape[0] < int(1.5 * n_samples):
                    t_tile, t_mask = generate_tile(patches, masks, tile_size, [1, 2])
                    r_tile = torch.cat([r_tile, t_tile], dim=0)
                    r_mask = torch.cat([r_mask, t_mask], dim=0)

                idx_perm = torch.randperm(r_tile.shape[0])
                idx_perm = idx_perm[:n_samples]
                tile_perm = r_tile[idx_perm, :]
                mask_perm = r_mask[idx_perm, :]

                # merge tiles
                box_1 = obj.xyxy
                [x_1, y_1, x_2, y_2] = box_1.type(torch.IntTensor)
                x_ref = x_init[:, :, y_1:y_2, x_1:x_2]
                x_new = ((1.0 - mask_perm) * x_ref) + mask_perm * (0.0 * x_ref + 1.0 * tile_perm)

                # randomly roll-back
                rand_rb = torch.rand([n_samples, 1, 1, 1], device=self.estimator.device)
                x_new = torch.where(rand_rb < 0.8, x_new, x_ref)
                x_cand[:, :, y_1:y_2, x_1:x_2] = x_new

            # spatial drop
            n_mask = drop_block2d(x_cand, 0.05, 3)
            x_cand = (1.0 - n_mask) * x_org + n_mask * x_cand
            # x_cand = smooth_image(x_cand, x_org, epsilon, 10)
            x_cand = torch.round(x_cand * 255.0) / 255.0
            x_cand = torch.clamp(x_cand, x_org - 2.5 * self.eps, x_org + 2.5 * self.eps)
            x_cand = torch.clamp(x_cand, 0.0, 1.0)

            # update results
            _, adv_position = self.collector(self.estimator, x_cand)
            for idx in range(n_samples):
                cur_position = adv_position[idx]

                for key, obj in tile_mat.items():

                    idx_i, idx_j = key
                    box_1 = obj.xyxy
                    obj_threshold = obj.threshold
                    [x_1, y_1, x_2, y_2] = box_1.type(torch.IntTensor)
                    overlay = bbox_ioa(box_1.type(torch.FloatTensor), cur_position.type(torch.FloatTensor))
                    bcount = torch.sum(overlay > 0.0).item()

                    x_ref = x_org[:, :, y_1:y_2, x_1:x_2]
                    x_cur = x_cand[idx, :, y_1:y_2, x_1:x_2].clone()

                    pert = x_cur - x_ref
                    loss = self._get_loss(pert, self.eps)
                    eligible = torch.max(torch.abs(pert)) < self.eps and bcount >= obj_threshold
                    tpatch_cur = TileObj(tile_size=tile_size, device=self.estimator.device)
                    tpatch_cur.update(eligible, bcount, torch.sum(loss), x_cur)
                    # insert op
                    prev = tile_mat[(idx_i, idx_j)]
                    prev.insert(tpatch_cur)
                    tile_mat[(idx_i, idx_j)] = prev

        # clean non-active regions
        x_out = x_init.clone()
        x_eval = self._assemble(tile_mat, x_org)
        _, adv_position = self.collector(self.estimator, x_eval)
        cur_position = adv_position[0]
        for key, obj in tile_mat.items():
            idx_i, idx_j = key
            box_1 = obj.xyxy
            [x_1, y_1, x_2, y_2] = box_1.type(torch.IntTensor)
            overlay = bbox_ioa(box_1.type(torch.FloatTensor), cur_position.type(torch.FloatTensor))
            bcount = torch.sum(overlay > 0.0).item()

            x_ref = x_init[:, :, y_1:y_2, x_1:x_2]
            x_tag = x_eval[:, :, y_1:y_2, x_1:x_2]
            cur_mask = torch.zeros_like(x_ref)
            if bcount > 1:
                bbox = cur_position[overlay > 0.0]
                for box in bbox:
                    bx1 = torch.clamp_min(box[0] - x_1, 0)
                    by1 = torch.clamp_min(box[1] - y_1, 0)
                    bx2 = torch.clamp_max(box[2] - x_1, (x_2 - x_1 - 1).to(self.estimator.device))
                    by2 = torch.clamp_max(box[3] - y_1, (y_2 - y_1 - 1).to(self.estimator.device))
                    cur_mask[:, :, by1:by2, bx1:bx2] = 1.0
            else:
                prev = tile_mat[(idx_i, idx_j)]
                prev.pop()
                tile_mat[(idx_i, idx_j)] = prev

            a_mask = drop_block2d(x_ref, 0.05, 1)
            cur_mask = cur_mask * a_mask
            updated = ((1.0 - cur_mask) * x_ref) + cur_mask * (0.0 * x_ref + 1.0 * x_tag)
            updated = ((1.0 - cur_mask) * x_ref) + cur_mask * (0.0 * x_ref + 1.0 * updated)

            x_out[:, :, y_1:y_2, x_1:x_2] = updated

        return x_out, tile_mat

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

        if not isinstance(self.candidates, list):
            raise TypeError("Candidates must be stored in list.")

        if len(self.candidates) < 1:
            raise ValueError("The list of candidates is empty.")
