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
Implementation of the adversarial texture attack on object trackers in PyTorch.

| Paper link: https://arxiv.org/abs/1904.11042
"""
import logging
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_tracking.object_tracker import ObjectTrackerMixin
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

logger = logging.getLogger(__name__)


class AdversarialTexturePyTorch(EvasionAttack):
    """
    Implementation of the adversarial texture attack on object trackers in PyTorch.

    | Paper link: https://arxiv.org/abs/1904.11042
    """

    attack_params = EvasionAttack.attack_params + [
        "patch_height",
        "patch_width",
        "xmin",
        "ymin",
        "step_size",
        "max_iter",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (ObjectTrackerMixin, LossGradientsMixin, BaseEstimator)

    def __init__(
        self,
        estimator,
        patch_height: int,
        patch_width: int,
        x_min: int = 0,
        y_min: int = 0,
        step_size: float = 1.0 / 255.0,
        max_iter: int = 500,
        batch_size: int = 16,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.AdversarialTexturePyTorch`.

        :param estimator: A trained estimator.
        :param patch_height: Height of patch.
        :param patch_width: Width of patch.
        :param x_min: Vertical position of patch, top-left corner.
        :param y_min: Horizontal position of patch, top-left corner.
        :param step_size: The step size.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        import torch

        super().__init__(estimator=estimator, summary_writer=summary_writer)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.x_min = x_min
        self.y_min = y_min
        self.step_size = step_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

        self._batch_id = 0
        self._i_max_iter = 0

        self.patch_shape = (self.patch_height, self.patch_width, 3)

        if self.estimator.channels_first:
            raise ValueError("Input shape has to be either NHWC or NFHWC.")

        self.i_h_patch = 0
        self.i_w_patch = 1

        self.i_h = 1
        self.i_w = 2

        if not (self.estimator.postprocessing_defences is None or self.estimator.postprocessing_defences == []):
            raise ValueError(
                "Framework-specific implementation of Adversarial Patch attack does not yet support "
                + "postprocessing defences."
            )

        mean_value = (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2.0 + self.estimator.clip_values[
            0
        ]
        self._initial_value = np.ones(self.patch_shape) * mean_value
        self._patch = torch.tensor(self._initial_value, requires_grad=True, device=self.estimator.device)

    def _train_step(
        self,
        videos: "torch.Tensor",
        target: List[Dict[str, "torch.Tensor"]],
        y_init: "torch.Tensor",
        foreground: Optional["torch.Tensor"],
        patch_points: Optional[np.ndarray],
    ) -> "torch.Tensor":
        """
        Apply a training step to the batch based on a mini-batch.

        :param videos: Video samples.
        :param target: Target labels/boxes.
        :param y_init: Initial labels/boxes.
        :param foreground: Foreground mask.
        :param patch_points: Array of shape (nb_frames, 4, 2) containing four pairs of integers (height, width)
                             corresponding to the coordinates of the four corners top-left, top-right, bottom-right,
                             bottom-left of the transformed image in the coordinate-system of the original image.
        :return: Loss.
        """
        import torch

        self.estimator.model.zero_grad()
        loss = self._loss(videos, target, y_init, foreground, patch_points)
        loss.backward(retain_graph=True)

        if self._patch.grad is not None:
            gradients = self._patch.grad.sign() * self.step_size
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                grad=np.expand_dims(self._patch.grad.detach().cpu().numpy(), axis=0),
                patch=None,
                estimator=None,
                x=None,
                y=None,
            )

        self._patch.grad = torch.zeros(
            self._patch.grad.shape, device=self._patch.grad.device, dtype=self._patch.grad.dtype
        )

        with torch.no_grad():
            self._patch[:] = torch.clamp(
                self._patch + gradients, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
            )

        return loss

    def _predictions(
        self,
        videos: "torch.Tensor",
        y_init: "torch.Tensor",
        foreground: Optional["torch.Tensor"],
        patch_points: Optional[np.ndarray],
    ) -> List[Dict[str, "torch.Tensor"]]:
        """
        Predict object tracking estimator on patched videos.

        :param videos: Video samples.
        :param y_init: Initial labels/boxes.
        :param foreground: Foreground mask.
        :param patch_points: Array of shape (nb_frames, 4, 2) containing four pairs of integers (height, width)
                             corresponding to the coordinates of the four corners top-left, top-right, bottom-right,
                             bottom-left of the transformed image in the coordinate-system of the original image.
        :return: Predicted labels/boxes.
        """
        import torch

        patched_input = self._apply_texture(videos, self._patch, foreground=foreground, patch_points=patch_points)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        predictions = self.estimator.predict(patched_input, y_init=y_init)  # pylint: disable=W0212

        return predictions

    def _loss(
        self,
        videos: "torch.Tensor",
        target: List[Dict[str, "torch.Tensor"]],
        y_init: "torch.Tensor",
        foreground: Optional["torch.Tensor"],
        patch_points: Optional[np.ndarray],
    ) -> "torch.Tensor":
        """
        Calculate L1-loss.

        :param videos: Video samples.
        :param target: Target labels/boxes.
        :param y_init: Initial labels/boxes.
        :param foreground: Foreground mask.
        :param patch_points: Array of shape (nb_frames, 4, 2) containing four pairs of integers (height, width)
                             corresponding to the coordinates of the four corners top-left, top-right, bottom-right,
                             bottom-left of the transformed image in the coordinate-system of the original image.
        :return: Loss.
        """
        import torch

        y_pred = self._predictions(videos, y_init, foreground, patch_points)
        loss = torch.nn.L1Loss(reduction="sum")(y_pred[0]["boxes"].float(), target[0]["boxes"].float())
        for i in range(1, len(y_pred)):
            loss = loss + torch.nn.L1Loss(reduction="sum")(y_pred[i]["boxes"].float(), target[i]["boxes"].float())

        return loss

    def _get_patch_mask(self, nb_samples: int) -> "torch.Tensor":
        """
        Create patch mask.

        :param nb_samples: Number of samples.
        :return: Patch mask.
        """
        import torch

        image_mask_np = np.ones((self.patch_height, self.patch_width))

        image_mask_np = np.expand_dims(image_mask_np, axis=2)
        image_mask_np = np.broadcast_to(image_mask_np, self.patch_shape)
        image_mask = torch.Tensor(np.array(image_mask_np)).to(self.estimator.device)
        image_mask = torch.stack([image_mask] * nb_samples, dim=0)
        return image_mask

    def _apply_texture(
        self,
        videos: "torch.Tensor",
        patch: "torch.Tensor",
        foreground: Optional["torch.Tensor"],
        patch_points: Optional[np.ndarray],
    ) -> "torch.Tensor":
        """
        Apply texture over background and overlay foreground.

        :param videos: Video samples.
        :param patch: Patch to apply.
        :param foreground: Foreground mask.
        :param patch_points: Array of shape (nb_frames, 4, 2) containing four pairs of integers (height, width)
                             corresponding to the coordinates of the four corners top-left, top-right, bottom-right,
                             bottom-left of the transformed image in the coordinate-system of the original image.
        :return: Patched videos.
        """
        import torch
        import torchvision

        nb_samples = videos.shape[0]
        nb_frames = videos.shape[1]
        frame_height = videos.shape[2]
        frame_width = videos.shape[3]

        image_mask = self._get_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        if patch_points is None:
            pad_h_before = self.x_min
            pad_h_after = int(videos.shape[self.i_h + 1] - pad_h_before - image_mask.shape[self.i_h_patch + 1])

            pad_w_before = self.y_min
            pad_w_after = int(videos.shape[self.i_w + 1] - pad_w_before - image_mask.shape[self.i_w_patch + 1])

            image_mask = image_mask.permute(0, 3, 1, 2)

            image_mask = torchvision.transforms.functional.pad(
                img=image_mask,
                padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
                fill=0,
                padding_mode="constant",
            )

            image_mask = image_mask.permute(0, 2, 3, 1)

            image_mask = torch.unsqueeze(image_mask, dim=1)
            image_mask = torch.repeat_interleave(image_mask, dim=1, repeats=nb_frames)
            image_mask = image_mask.float()

            padded_patch = padded_patch.permute(0, 3, 1, 2)

            padded_patch = torchvision.transforms.functional.pad(
                img=padded_patch,
                padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
                fill=0,
                padding_mode="constant",
            )

            padded_patch = padded_patch.permute(0, 2, 3, 1)

            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            padded_patch = torch.repeat_interleave(padded_patch, dim=1, repeats=nb_frames)

            padded_patch = padded_patch.float()

        else:

            startpoints = [[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]]
            endpoints = np.zeros_like(patch_points)
            endpoints[:, :, 0] = patch_points[:, :, 1]
            endpoints[:, :, 1] = patch_points[:, :, 0]

            image_mask = image_mask.permute(0, 3, 1, 2)

            image_mask = torchvision.transforms.functional.resize(
                img=image_mask,
                size=[int(videos.shape[2]), int(videos.shape[3])],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

            image_mask_list = []

            for i_frame in range(nb_frames):

                image_mask_i = torchvision.transforms.functional.perspective(
                    img=image_mask,
                    startpoints=startpoints,
                    endpoints=endpoints[i_frame],
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )

                image_mask_i = image_mask_i.permute(0, 2, 3, 1)

                image_mask_list.append(image_mask_i)

            image_mask = torch.stack(image_mask_list, dim=1)
            image_mask = image_mask.float()

            padded_patch = padded_patch.permute(0, 3, 1, 2)

            padded_patch = torchvision.transforms.functional.resize(
                img=padded_patch,
                size=[int(videos.shape[2]), int(videos.shape[3])],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

            padded_patch_list = []

            for i_frame in range(nb_frames):
                padded_patch_i = torchvision.transforms.functional.perspective(
                    img=padded_patch,
                    startpoints=startpoints,
                    endpoints=endpoints[i_frame],
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )

                padded_patch_i = padded_patch_i.permute(0, 2, 3, 1)

                padded_patch_list.append(padded_patch_i)

            padded_patch = torch.stack(padded_patch_list, dim=1)
            padded_patch = padded_patch.float()

        inverted_mask = (
            torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)).to(self.estimator.device) - image_mask
        )

        if foreground is not None:
            combined = (
                videos * inverted_mask
                + padded_patch * image_mask
                - padded_patch * ~foreground.bool()
                + videos * ~foreground.bool() * image_mask
            )
        else:
            combined = videos * inverted_mask + padded_patch * image_mask

        return combined

    def generate(  # type: ignore  # pylint: disable=W0222
        self, x: np.ndarray, y: List[Dict[str, np.ndarray]], **kwargs
    ) -> np.ndarray:
        """
        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: Input videos of shape NFHWC.
        :param y: True labels of format `List[Dict[str, np.ndarray]]`, one dictionary for each input image. The keys of
                  the dictionary are:

                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                         0 <= y1 < y2 <= H.

        :Keyword Arguments:
            * *shuffle* (``np.ndarray``) --
              Shuffle order of samples, labels, initial boxes, and foregrounds for texture generation.
            * *y_init* (``np.ndarray``) --
              Initial boxes around object to be tracked of shape (nb_samples, 4) with second dimension representing
              [x1, y1, x2, y2] with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
            * *foreground* (``np.ndarray``) --
              Foreground masks of shape NFHWC of boolean values with False/0.0 representing foreground, preventing
              updates to the texture, and True/1.0 for background, allowing updates to the texture.
            * *patch_points* (``np.ndarray``) --
              Array of shape (nb_frames, 4, 2) containing four pairs of integers (height, width) corresponding to the
              four corners top-left, top-right, bottom-right, bottom-left of the transformed image in the coordinates
              of the original image.

        :return: An array with images patched with adversarial texture.
        """
        import torch

        shuffle = kwargs.get("shuffle", True)
        y_init = kwargs.get("y_init")
        foreground = kwargs.get("foreground")
        if foreground is None:
            foreground = np.ones_like(x)
        patch_points = kwargs.get("patch_points")

        class TrackingDataset(torch.utils.data.Dataset):
            """
            Object tracking dataset in PyTorch.
            """

            def __init__(self, x, y, y_init, foreground):
                self.x = x
                self.y = y
                self.y_init = y_init
                self.foreground = foreground

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                img = self.x[idx]

                target = {}
                target["boxes"] = torch.from_numpy(y[idx]["boxes"])

                y_init_i = self.y_init[idx]
                foreground_i = self.foreground[idx]

                return img, target, y_init_i, foreground_i

        dataset = TrackingDataset(x, y, y_init, foreground)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

        for i_max_iter in trange(self.max_iter, desc="Adversarial Texture PyTorch", disable=not self.verbose):

            self._i_max_iter = i_max_iter
            self._batch_id = 0

            for videos_i, target_i, y_init_i, foreground_i in data_loader:
                self._batch_id += 1
                videos_i = videos_i.to(self.estimator.device)
                y_init_i = y_init_i.to(self.estimator.device)
                foreground_i = foreground_i.to(self.estimator.device)
                target_i_list = []
                for i_t in range(videos_i.shape[0]):
                    target_i_list.append({"boxes": target_i["boxes"][i_t].to(self.estimator.device)})

                _ = self._train_step(
                    videos=videos_i,
                    target=target_i_list,
                    y_init=y_init_i,
                    foreground=foreground_i,
                    patch_points=patch_points,
                )

                # Write summary
                if self.summary_writer is not None:  # pragma: no cover
                    self.summary_writer.update(
                        batch_id=self._batch_id,
                        global_step=self._i_max_iter,
                        grad=None,
                        patch=self._patch.detach().cpu().numpy(),
                        estimator=self.estimator,
                        x=videos_i.detach().cpu().numpy(),
                        y=target_i_list,
                    )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return self.apply_patch(x=x, foreground=foreground, patch_points=patch_points)

    def apply_patch(
        self,
        x: np.ndarray,
        patch_external: Optional[np.ndarray] = None,
        foreground: Optional[np.ndarray] = None,
        patch_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        A function to apply the learned adversarial texture to videos.

        :param x: Videos of shape NFHWC to apply adversarial texture.
        :param patch_external: External patch to apply to videos `x`.
        :param foreground: Foreground masks of shape NFHWC of boolean values with False/0.0 representing foreground,
                           preventing updates to the texture, and True/1.0 for background, allowing updates to the
                           texture.
        :param patch_points: Array of shape (nb_frames, 4, 2) containing four pairs of integers (height, width)
                             corresponding to the coordinates of the four corners top-left, top-right, bottom-right,
                             bottom-left of the transformed image in the coordinate-system of the original image.
        :return: The videos with adversarial textures.
        """
        import torch

        patch_tensor = (
            torch.Tensor(patch_external).to(self.estimator.device) if patch_external is not None else self._patch
        )
        x_tensor = torch.Tensor(x).to(self.estimator.device)
        if foreground is None:
            foreground_tensor = None
        else:
            foreground_tensor = torch.Tensor(foreground).to(self.estimator.device)

        return (
            self._apply_texture(
                videos=x_tensor, patch=patch_tensor, foreground=foreground_tensor, patch_points=patch_points
            )
            .detach()
            .cpu()
            .numpy()
        )

    def reset_patch(self, initial_patch_value: Optional[Union[float, np.ndarray]] = None) -> None:
        """
        Reset the adversarial texture.

        :param initial_patch_value: Patch value to use for resetting the patch.
        """
        import torch

        if initial_patch_value is None:
            self._patch.data = torch.Tensor(self._initial_value).double()
        elif isinstance(initial_patch_value, float):
            initial_value = np.ones(self.patch_shape) * initial_patch_value
            self._patch.data = torch.Tensor(initial_value).double()
        elif self._patch.shape == initial_patch_value.shape:
            self._patch.data = torch.Tensor(initial_patch_value).double()
        else:
            raise ValueError("Unexpected value for initial_patch_value.")

    def _check_params(self) -> None:

        if not isinstance(self.patch_height, int) or self.patch_height <= 0:
            raise ValueError("The patch height `patch_height` has to be of type int and larger than zero.")

        if not isinstance(self.patch_width, int) or self.patch_width <= 0:
            raise ValueError("The patch width `patch_width` has to be of type int and larger than zero.")

        if not isinstance(self.x_min, int) or self.x_min < 0:
            raise ValueError("The vertical position `x_min` has to be of type int and larger than zero.")

        if not isinstance(self.y_min, int) or self.y_min < 0:
            raise ValueError("The horizontal position `y_min` has to be of type int and larger than zero.")

        if not isinstance(self.step_size, float) or self.step_size <= 0:
            raise ValueError("The step size `step_size` has to be of type float and larger than zero.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be of type int and larger than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be of type int and larger than zero.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
