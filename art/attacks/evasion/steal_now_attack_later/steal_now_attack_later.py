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
This module implements the paper: "Steal Now and Attack Later: Evaluating Robustness of Object Detection against Black-box Adversarial Attacks"

| Paper link: https://arxiv.org/abs/2304.05370
"""

# pylint: disable=C0302

import logging
import random
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.steal_now_attack_later.bbox_ioa import bbox_ioa
from art.attacks.evasion.steal_now_attack_later.drop_block2d import drop_block2d

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

logger = logging.getLogger(__name__)

# collection
def collect_patches_from_images(model: "torch.nn.Module",
                                imgs: "torch.Tensor") -> Tuple[list, list]:
    """
    Collect patches and corrsponding spatial information by the model from images.

    :param model: Object detection model.
    :param imgs: Target images.

    :return: Detected objects and corrsponding spatial information.
    """
    import torch

    bs = imgs.shape[0]
    y = model.inference(imgs)

    candidates_patch = []
    candidates_position = []
    for i in range(bs):
        patch = []
        if y[i].shape[0] == 0:
            candidates_patch.append(patch)
            candidates_position.append(torch.zeros((0, 4), device=model.device))
            continue

        pos_matrix = y[i][:, :4].clone().int()
        pos_matrix[:, 0] = torch.clamp_min(pos_matrix[:, 0], 0)
        pos_matrix[:, 1] = torch.clamp_min(pos_matrix[:, 1], 0)
        pos_matrix[:, 2] = torch.clamp_max(pos_matrix[:, 2], imgs.shape[3])
        pos_matrix[:, 3] = torch.clamp_max(pos_matrix[:, 3], imgs.shape[2])
        for e in pos_matrix:
            p = imgs[i, :, e[1]:e[3], e[0]:e[2]]
            patch.append(p.to(model.device))

        candidates_patch.append(patch)
        candidates_position.append(pos_matrix)

    return candidates_patch, candidates_position

# tiling
def _generate_tile_kernel(patch: list,
                          mask: list,
                          tile_size: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
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
    w, h = t_patch.shape[-2], t_patch.shape[-1]
    boundary = 1
    tile_size = max(tile_size - 2 * boundary, 1)

    if h > w:
        flip = True
        FlipOp = torchvision.transforms.RandomVerticalFlip(0.2)
        max_len = h
        min_len = w
        t_patch= torch.permute(t_patch, (0, 2, 1))
        t_mask= torch.permute(t_mask, (0, 2, 1))
    else:
        flip = False
        FlipOp = torchvision.transforms.RandomHorizontalFlip(0.2)
        max_len = w
        min_len = h

    if max_len > tile_size:
        s = tile_size / max_len
        new_len = round(min_len * s)
        p1 = torchvision.transforms.Resize((tile_size, new_len))(t_patch)
        # fix for the case that (strides - new_len) > new_len
        p_list = []

        for _ in range(tile_size // new_len):
            p_list.append(FlipOp(p1))

        p2 = torchvision.transforms.RandomCrop((tile_size, tile_size % new_len))(p1)
        p_list.append(FlipOp(p2))

        pp = torch.cat(p_list, dim=-1)
        pp = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(pp)
        mm = torch.where(pp == 0, torch.zeros_like(pp), torch.ones_like(pp))

    elif max_len >= tile_size / 2.0:
        s = (tile_size / 2.0) / max_len
        new_len = round(min_len * s)

        p_list = []
        for _ in range(tile_size // new_len):
            repeat = 2
            p1_list = []
            for _ in range(repeat):
                p1 = torchvision.transforms.Resize((tile_size // 2, new_len))(t_patch)
                if torch.rand([]) < 0.6:
                    p1_list.append(FlipOp(p1))
                else:
                    p1_list.append(torch.zeros_like(p1))
            p1 = torch.cat(p1_list, dim=-2)
            p_list.append(p1)

        p2 = torchvision.transforms.RandomCrop((tile_size, tile_size % new_len))(p1)
        p_list.append(FlipOp(p2))

        pp = torch.cat(p_list, dim=-1)
        pp = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(pp)
        mm = torch.where(pp == 0, torch.zeros_like(pp), torch.ones_like(pp))

    else:
        t = torch.cat([t_patch[None, :], t_mask[None, :]], dim=0)
        pp = []
        mm = []
        for _ in range(tile_size // min_len):
            p1_list = []
            m1_list = []
            for _ in range(tile_size // max_len):
                if torch.rand([]) < 0.4:
                    t = FlipOp(t)
                    p1_list.append(t[0, :])
                    m1_list.append(t[1, :])
                else:
                    p1_list.append(torch.zeros_like(t_patch))
                    m1_list.append(torch.zeros_like(t_mask))
            p1 = torch.cat(p1_list, dim=-2)
            m1 = torch.cat(m1_list, dim=-2)
            pp.append(p1)
            mm.append(m1)
        pp = torch.cat(pp, dim=-1)
        mm = torch.cat(mm, dim=-1)
        pp = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(pp)
        mm = torchvision.transforms.CenterCrop((tile_size + 2 * boundary, tile_size + 2 * boundary))(mm)

    if flip:
        pp= torch.permute(pp, (0, 2, 1))
        mask= torch.permute(mm, (0, 2, 1))
    else:
        mask = mm.clone()

    return pp, mask

def generate_tile(patches: list,
                  masks: list,
                  tile_size: int,
                  scale: list) -> Tuple["torch.Tensor", "torch.Tensor"]:
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
    for s in scale:
        cur_strides = tile_size // s
        cur_tile = []
        cur_mask = []

        for x_i in range(s):
            t1_list = []
            m1_list = []
            for y_i in range(s):
                g_tile, f_mask = _generate_tile_kernel(patches, masks, tile_size=cur_strides)
                t1_list.append(g_tile[None, :])
                m1_list.append(f_mask[None, :])
            t1 = torch.cat(t1_list, dim=-2)
            m1 = torch.cat(m1_list, dim=-2)
            cur_tile.append(t1)
            cur_mask.append(m1)
        cur_tile = torch.cat(cur_tile, dim=-1)
        cur_mask = torch.cat(cur_mask, dim=-1)

        tile = torch.cat([tile, cur_tile], dim=0)
        mask = torch.cat([mask, cur_mask], dim=0)

    return tile, mask

# internal used
class TileObj:
    def __init__(self,
                 tile_size: int,
                 device: "torch.cuda.device") -> None:
        import torch

        self.patch = torch.zeros((3, tile_size, tile_size), device = device)
        self.diff = torch.ones([], device = device) * self.patch.shape.numel()
        self.bcount = 0
        self.eligible = False

    def update(self,
               eligible = None,
               bcount = None,
               diff = None,
               patch = None
               ) -> None:

        if not (eligible is None):
            self.eligible = eligible

        if not (bcount is None):
            self.bcount = bcount

        if not (diff is None):
            self.diff = diff

        if not (patch is None):
            self.patch = patch

    def compare(self, target: "TileObj") -> bool:
        if self.eligible == True and target.eligible == False:
            return True
        elif self.eligible == False and target.eligible == True:
            return False
        else:
            if self.bcount > target.bcount:
                return True
            elif self.bcount < target.bcount:
                return False
            else:
                if self.diff < target.diff:
                    return True
                else:
                    return False


class TileArray:
    def __init__(self,
                 xyxy: list, 
                 threshold: int, 
                 tile_size: int, 
                 k: int, 
                 device: "torch.cuda.device") -> None:
        import torch

        self.threshold = threshold
        self.strides = tile_size
        self.device = device
        self.xyxy = torch.Tensor(xyxy).to(device)
        self.k = k
        self.patch_list = [TileObj(tile_size=tile_size, device=device)] * self.k

    def insert(self, target: TileObj) -> None:

        if target.bcount < self.threshold:
            return

        prev = self.patch_list
        out = []
        for k_it in range(self.k):
            if target.compare(prev[k_it]):
                out.append(target)
                out = out + prev[k_it:]
                break
            else:
                out.append(prev[k_it])

        self.patch_list = out[:self.k]

    def pop(self) -> None:
        out = self.patch_list[1:] + [TileObj(tile_size=self.strides, device=self.device)]
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
        estimator: "torch.nn.Module",
        eps: float,
        max_iter: int,
        num_grid: int,
    ) -> None:
        """
        Create a SNAL attack instance.

        :param estimator: A trained YOLOv8 model or other models with the same output format
        :param eps: Maximum perturbation that the attacker can introduce.
        :param max_iter: The maximum number of iterations.
        :param num_grid: The number of grids for width and high dimension.
        """
        super().__init__(estimator=estimator)
        self.eps = eps
        self.max_iter = max_iter
        self.num_grid = num_grid
        self.batch_size = 1
        self.candidates = None
        self.threshold_objs = 1 # the expect number of objects
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
            batch_index_1 =  batch_id * self.batch_size
            batch_index_2 = min((batch_id + 1) * self.batch_size, x_adv.shape[0])
            x_batch = x_adv[batch_index_1:batch_index_2]
            x_adv[batch_index_1:batch_index_2] = self._generate_batch(x_batch)

        return x_adv

    def _generate_batch(self, x_batch: np.ndarray, y_batch: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run the attack on a batch of images.

        :param x_batch: A batch of original examples.
        :param y_batch: Not Used.
        :return: A batch of adversarial examples.
        """
        import torch

        x_org = torch.from_numpy(x_batch).to(self.estimator.model.device)
        x_adv = x_org.clone()

        cond = torch.logical_or(x_org < 0.0, x_org > 1.0)
        if torch.any(cond):
            raise ValueError("The value of each pixel must be normalized in the range [0, 1].")

        x_adv = self._attack(x_adv, x_org)

        return x_adv.cpu().detach().numpy()

    def set_candidates(self, candidates: list) -> None:
        """
        Assign candidates that will be used to generate perturbations during the attack.
        """
        self.candidates = candidates

    def _attack(self,
                x_adv: "torch.Tensor",
                x: "torch.Tensor") -> "torch.Tensor":
        """
        Run attack.

        :param x_batch: A batch of original examples.
        :param y_batch: Not Used.
        :return: A batch of adversarial examples.
        """
        import torch
        import torchvision

        self.estimator.count_reset()
        if self.candidates is None:
            raise ValueError("A set of patches should be collected before executing the attack.")

        if x.shape[-1] % self.num_grid != 0 or \
           x.shape[-2] % self.num_grid != 0:
            raise ValueError("The size of the image must be divided by the number of grids")
        tile_size = x.shape[-1] // self.num_grid

        # Prapare a 2D array to store the results of each grid
        buffer_depth = 5
        tile_mat = {}
        for ii in range(self.num_grid):
            for jj in range(self.num_grid):
                x1 = ii * tile_size
                y1 = jj * tile_size
                x2 = x1 + tile_size
                y2 = y1 + tile_size
                tile_mat[(ii,jj)] = TileArray(list([x1, y1, x2, y2]),
                                              self.threshold_objs,
                                              tile_size,
                                              buffer_depth,
                                              self.estimator.model.device)

        # init guess
        n_samples = 10
        x_adv, tile_mat = self._init_guess(tile_mat,
                                           x_adv,
                                           x,
                                           tile_size,
                                           n_samples=n_samples)


        b = 0
        candidates_patch = self.candidates
        candidates_mask = [None] * len(candidates_patch)

        r_tile = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.model.device)
        r_mask = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.model.device)
        while r_tile.shape[0] < n_samples:
            t_tile, t_mask = generate_tile(candidates_patch,
                                           candidates_mask,
                                           tile_size,
                                           [1, 2])
            r_tile = torch.cat([r_tile, t_tile], dim=0)
            r_mask = torch.cat([r_mask, t_mask], dim=0)


        for _ in range(self.max_iter):
            adv_patch, adv_position = collect_patches_from_images(self.estimator, x_adv)
            adv_position = adv_position[0]
            candidates_patch = candidates_patch + adv_patch[0]
            candidates_mask = candidates_mask + [None] * len(adv_patch[0])

            for key, obj in tile_mat.items():
                ii, jj = key
                b1 = obj.xyxy
                obj_threshold = obj.threshold
                [x1, y1, x2, y2] = b1.type(torch.IntTensor)
                overlay = bbox_ioa(b1.type(torch.FloatTensor), adv_position.type(torch.FloatTensor))
                bcount = torch.sum(overlay > 0.0).item()

                pert = x_adv[b, :, y1:y2, x1:x2] - x[b, :, y1:y2, x1:x2]
                loss = self._get_loss(pert, self.eps)
                eligible = torch.max(torch.abs(pert)) < self.eps and bcount >= obj_threshold
                TPatch_cur = TileObj(tile_size=tile_size, device=self.estimator.model.device)
                TPatch_cur.update(eligible, bcount, torch.sum(loss), x_adv[b, :, y1:y2, x1:x2].clone())

                # insert op
                prev = tile_mat[(ii, jj)]
                prev.insert(TPatch_cur)
                tile_mat[(ii, jj)] = prev

                sorted_patch = tile_mat[(ii,jj)].patch_list
                bcount_list = []
                for sp in sorted_patch:
                    if sp.bcount >= obj_threshold:
                        bcount_list.append(sp)

                if len(bcount_list) == buffer_depth and bcount_list[-1].bcount > obj_threshold:
                    tile_mat[(ii, jj)].threshold = obj_threshold + 1

                if len(bcount_list) < buffer_depth:

                    while r_tile.shape[0] < int( 1.5 * n_samples):
                        t_tile, t_mask = generate_tile(candidates_patch,
                                                       candidates_mask,
                                                       tile_size,
                                                       [1, 2])
                        r_tile = torch.cat([r_tile, t_tile], dim=0)
                        r_mask = torch.cat([r_mask, t_mask], dim=0)

                    # select n_sample candidates
                    c_tile = r_tile
                    idx_perm = torch.randperm(c_tile.shape[0])
                    idx_perm = idx_perm[:n_samples]
                    c_tile = r_tile[idx_perm, :]
                    c_mask = r_mask[idx_perm, :]
                    x_ref = x[:, :, y1:y2, x1:x2]

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
                    x_ref = x[b, :, y1:y2, x1:x2]
                    updated = self._color_projection(target, x_ref, self.eps)

                x_adv[b, :, y1:y2, x1:x2] = updated
                x_adv = torch.round(x_adv * 255.0) / 255.0
                x_adv = torch.clamp(x_adv, x - 2.5 * self.eps, x + 2.5 * self.eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    def _get_loss(self,
                  pert: "torch.tensor",
                  epsilon: float) -> "torch.tensor":
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

    def _color_projection(self,
                          tile: "torch.tensor",
                          x_ref: "torch.tensor",
                          epsilon: float) -> "torch.tensor":
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
        mean_s = torch.mean(x_ref, dim=(-2,-1), keepdim=True)
        mean_t = torch.mean(x_ref, dim=(-2,-1), keepdim=True)
        std_s  = torch.std(set2 , dim=(-2,-1), keepdim=True)
        std_t  = torch.std(set2 , dim=(-2,-1), keepdim=True)
        scale = std_s / std_t
        set2 = (set2 - mean_t) * scale + mean_s
        set2 = torch.clamp(set2, 0.0, 1.0)
        
        set2 = set2 + sign * epsilon * scale
        set2 = torch.clamp(set2, 0, 1)

        updated = torch.where(cond, set1, set2)

        return updated

    def _assemble(self,
                  tile_mat: dict,
                  x_org: "torch.tensor") -> "torch.tensor":
        """
        Combine the best patches from each grid into a single image.

        :param tile_mat: Internal structure used to store patches for each mesh.
        :param x_org: The original images.
        :return: Perturbed images.
        """
        import torch

        ans = x_org.clone()
        for obj in tile_mat.values():
            [x1, y1, x2, y2] = obj.xyxy.type(torch.IntTensor)
            tile = obj.patch_list[0].patch[None, :]
            mask = torch.where(tile != 0, torch.ones_like(tile), torch.zeros_like(tile))
            ans[0, :, y1:y2, x1:x2] = mask * tile + (1.0 - mask) * ans[0, :, y1:y2, x1:x2]
        return ans

    def _init_guess(self,
                    tile_mat: dict,
                    x_init: "torch.tensor",
                    x_org: "torch.tensor",
                    tile_size: int,
                    n_samples: int) -> Tuple["torch.tensor", dict]:
        """
        Generate an initial perturbation for each grid.

        :param tile_mat: Internal structure used to store patches for each mesh.
        :param x_init: Perturbed images from previous runs.
        :param x_org: The original images.
        :param tile_size: The size of each tile.
        :return: Guessed images and internal structure.
        """
        import torch

        TRIAL = 10
        patches = self.candidates
        masks = [None] * len(self.candidates)
        for _ in range(TRIAL):
            x_cand = torch.zeros((n_samples, 3, x_init.shape[-2], x_init.shape[-1]),
                                  dtype=x_init.dtype,
                                  device=self.estimator.model.device)

            # generate tiles
            # To save the computing time, we generate some tiles in advance.
            # partial tiles are updated on-the-fly
            r_tile = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.model.device)
            r_mask = torch.zeros((0, 3, tile_size, tile_size), device=self.estimator.model.device)
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
                b1 = obj.xyxy
                [x1, y1, x2, y2] = b1.type(torch.IntTensor)
                x_ref = x_init[:, :, y1:y2, x1:x2]
                x_new = ((1.0 - mask_perm) * x_ref) + mask_perm * ( 0.0 * x_ref + 1.0 * tile_perm)

                # randomly roll-back
                rand_rb = torch.rand([n_samples, 1, 1, 1], device=self.estimator.model.device)
                x_new = torch.where(rand_rb < 0.8, x_new, x_ref)
                x_cand[:, :, y1:y2, x1:x2] = x_new

            # spatial drop
            n_mask = drop_block2d(x_cand, 0.05, 3)
            x_cand = (1.0 - n_mask) * x_org + n_mask * x_cand
            #x_cand = smooth_image(x_cand, x_org, epsilon, 10)
            x_cand = torch.round(x_cand * 255.0) / 255.0
            x_cand = torch.clamp(x_cand, x_org - 2.5 * self.eps, x_org + 2.5 * self.eps)
            x_cand = torch.clamp(x_cand, 0.0, 1.0)

            # update results
            adv_patch, adv_position = collect_patches_from_images(self.estimator, x_cand)
            for idx in range(n_samples):
                cur_position = adv_position[idx]

                for key, obj in tile_mat.items():

                    ii, jj = key
                    b1 = obj.xyxy
                    obj_threshold = obj.threshold
                    [x1, y1, x2, y2] = b1.type(torch.IntTensor)
                    overlay = bbox_ioa(b1.type(torch.FloatTensor), cur_position.type(torch.FloatTensor))
                    bcount = torch.sum(overlay > 0.0).item()

                    x_ref = x_org[:, :, y1:y2, x1:x2]
                    x_cur = x_cand[idx, :, y1:y2, x1:x2].clone()

                    pert = x_cur - x_ref
                    loss = self._get_loss(pert, self.eps)
                    eligible = torch.max(torch.abs(pert)) < self.eps and bcount >= obj_threshold
                    TPatch_cur = TileObj(tile_size=tile_size, device=self.estimator.model.device)
                    TPatch_cur.update(eligible, bcount, torch.sum(loss), x_cur)
                    # insert op
                    prev = tile_mat[(ii, jj)]
                    prev.insert(TPatch_cur)
                    tile_mat[(ii, jj)] = prev

        # clean non-active regions
        x_out = x_init.clone()
        x_eval = self._assemble(tile_mat, x_org)
        adv_patch, adv_position = collect_patches_from_images(self.estimator, x_eval)
        cur_patch = adv_patch[0]
        cur_position = adv_position[0]
        for key, obj in tile_mat.items():
            ii, jj = key
            b1 = obj.xyxy
            [x1, y1, x2, y2] = b1.type(torch.IntTensor)
            overlay = bbox_ioa(b1.type(torch.FloatTensor), cur_position.type(torch.FloatTensor))
            bcount = torch.sum(overlay > 0.0).item()

            x_ref = x_init[:, :, y1:y2, x1:x2]
            x_tag = x_eval[:, :, y1:y2, x1:x2]
            cur_mask = torch.zeros_like(x_ref)
            if bcount > 1:
                bbox = cur_position[overlay > 0.0]
                for b in bbox:
                    bx1 = torch.clamp_min(b[0] - x1, 0)
                    by1 = torch.clamp_min(b[1] - y1, 0)
                    bx2 = torch.clamp_max(b[2] - x1, (x2 - x1 - 1).to(self.estimator.model.device))
                    by2 = torch.clamp_max(b[3] - y1, (y2 - y1 - 1).to(self.estimator.model.device))
                    cur_mask[:, :, by1:by2, bx1:bx2] = 1.0
            else:
                prev = tile_mat[(ii, jj)]
                prev.pop()
                tile_mat[(ii, jj)] = prev

            a_mask = drop_block2d(x_ref, 0.05, 1)
            cur_mask = cur_mask * a_mask
            updated = ((1.0 - cur_mask)  * x_ref) + cur_mask * ( 0.0 * x_ref + 1.0 * x_tag)
            updated = ((1.0 - cur_mask)  * x_ref) + cur_mask * ( 0.0 * x_ref + 1.0 * updated)

            x_out[:, :, y1:y2, x1:x2] = updated

        return x_out, tile_mat
