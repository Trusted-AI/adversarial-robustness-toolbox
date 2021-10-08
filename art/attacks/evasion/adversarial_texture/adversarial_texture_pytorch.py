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
Implementation of the adversarial patch attack for square and rectangular images and videos in PyTorch.

| Paper link: https://arxiv.org/abs/1712.09665
"""
import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.adversarial_patch.utils import insert_transformed_patch
from art.estimators.estimator import BaseEstimator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

logger = logging.getLogger(__name__)


class AdversarialTexturePyTorch(EvasionAttack):
    """
    Implementation of the adversarial patch attack for square and rectangular images and videos in PyTorch.

    | Paper link: https://arxiv.org/abs/1712.09665
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

    _estimator_requirements = (BaseEstimator,)

    def __init__(
        self,
        estimator,
        patch_height: int = 0,
        patch_width: int = 0,
        x_min: int = 0,
        y_min: int = 0,
        step_size: float = 1.0 / 255.0,
        max_iter: int = 500,
        batch_size: int = 16,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.AdversarialTexturePyTorch`.

        :param estimator: A trained estimator.
        :param patch_height: Height of patch.
        :param patch_width: Width of patch.
        :param x_min: Height of patch.
        :param y_min: Width of patch.
        :param step_size: The step size.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param verbose: Show progress bars.
        """
        import torch  # lgtm [py/repeated-import]

        super().__init__(estimator=estimator)
        self.step_size = step_size
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.x_min = x_min
        self.y_min = y_min

        self.image_shape = estimator.input_shape
        self.input_shape = self.estimator.input_shape

        self.patch_shape = (self.patch_height, self.patch_width, 3)

        self.verbose = verbose
        self._check_params()

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
        # self._patch = torch.from_numpy(self._initial_value)
        # self._patch.requires_grad = True
        # self._patch.to(self.estimator.device)

    def _train_step(
        self, images: "torch.Tensor", target: "torch.Tensor", y_init, foreground: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        self.estimator.model.zero_grad()
        loss = self._loss(images, target, y_init, foreground)
        loss.backward(retain_graph=True)

        gradients = self._patch.grad.sign() * self.step_size

        with torch.no_grad():
            self._patch[:] = torch.clamp(
                self._patch + gradients, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
            )

        return loss

    def _predictions(self, images: "torch.Tensor", y_init, foreground) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, self._patch, foreground=foreground)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        predictions = self.estimator.predict(patched_input, y_init=y_init)  # pylint: disable=W0212

        return predictions

    def _loss(
        self, images: "torch.Tensor", target: "torch.Tensor", y_init, foreground: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        y_pred = self._predictions(images, y_init, foreground)
        loss = torch.nn.L1Loss(size_average=False)(y_pred[0]["boxes"].float(), target["boxes"][0].float())

        return loss

    def _get_circular_patch_mask(self, nb_samples: int) -> "torch.Tensor":
        """
        Return a circular patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        image_mask = np.ones((self.patch_height, self.patch_width))

        image_mask = np.expand_dims(image_mask, axis=2)
        image_mask = np.broadcast_to(image_mask, self.patch_shape)
        image_mask = torch.Tensor(np.array(image_mask))
        image_mask = torch.stack([image_mask] * nb_samples, dim=0)
        return image_mask

    def _random_overlay(self, images: "torch.Tensor", patch: "torch.Tensor", foreground=None) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]
        import torchvision

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        pad_h_before = self.x_min
        pad_h_after = int(images.shape[self.i_h + 1] - pad_h_before - image_mask.shape[self.i_h_patch + 1])

        pad_w_before = self.y_min
        pad_w_after = int(images.shape[self.i_w + 1] - pad_w_before - image_mask.shape[self.i_w_patch + 1])

        image_mask = image_mask.permute(0, 3, 1, 2)

        image_mask = torchvision.transforms.functional.pad(
            img=image_mask,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        image_mask = image_mask.permute(0, 2, 3, 1)

        image_mask = torch.unsqueeze(image_mask, dim=1)
        image_mask = torch.repeat_interleave(image_mask, dim=1, repeats=images.shape[1])

        image_mask = image_mask.float()

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        padded_patch = padded_patch.permute(0, 3, 1, 2)

        padded_patch = torchvision.transforms.functional.pad(
            img=padded_patch,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        padded_patch = padded_patch.permute(0, 2, 3, 1)

        padded_patch = torch.unsqueeze(padded_patch, dim=1)
        padded_patch = torch.repeat_interleave(padded_patch, dim=1, repeats=images.shape[1])

        padded_patch = padded_patch.float()

        inverted_mask = torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)) - image_mask

        combined = (
            images * inverted_mask
            + padded_patch * image_mask
            - padded_patch * ~foreground.bool()
            + images * ~foreground.bool() * image_mask
        )

        return combined

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: An array with the original input images of shape NHWC or input videos of shape NFHWC.
        :param y: An array with the original true labels.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :return: An array with adversarial patch and an array of the patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        shuffle = kwargs.get("shuffle", True)
        y_init = kwargs.get("y_init")
        foreground = kwargs.get("foreground")

        class TrackingDataset(torch.utils.data.Dataset):
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
                target["labels"] = y[idx]["labels"]

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

        for _ in trange(self.max_iter, desc="Adversarial Texture PyTorch", disable=not self.verbose):
            for images_i, target_i, y_init_i, foreground_i in data_loader:
                _ = self._train_step(
                    images=images_i, target=target_i, y_init=y_init_i, foreground=foreground_i
                )

        return self.apply_patch(x=x, foreground=foreground)

    def apply_patch(
        self,
        x: np.ndarray,
        patch_external: Optional[np.ndarray] = None,
        foreground: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        A function to apply the learned adversarial patch to images or videos.

        :param x: Instances to apply randomly transformed patch.
        :param patch_external: External patch to apply to images `x`.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched samples.
        """
        import torch  # lgtm [py/repeated-import]

        patch = patch_external if patch_external is not None else self._patch
        x = torch.Tensor(x)
        foreground = torch.Tensor(foreground)

        return self._random_overlay(images=x, patch=patch, foreground=foreground).detach().cpu().numpy()

    def reset_patch(self, initial_patch_value: Optional[Union[float, np.ndarray]] = None) -> None:
        """
        Reset the adversarial patch.

        :param initial_patch_value: Patch value to use for resetting the patch.
        """
        import torch  # lgtm [py/repeated-import]

        if initial_patch_value is None:
            self._patch.data = torch.Tensor(self._initial_value).double()
        elif isinstance(initial_patch_value, float):
            initial_value = np.ones(self.patch_shape) * initial_patch_value
            self._patch.data = torch.Tensor(initial_value).double()
        elif self._patch.shape == initial_patch_value.shape:
            self._patch.data = torch.Tensor(initial_patch_value).double()
        else:
            raise ValueError("Unexpected value for initial_patch_value.")

    @staticmethod
    def insert_transformed_patch(x: np.ndarray, patch: np.ndarray, image_coords: np.ndarray):
        """
        Insert patch to image based on given or selected coordinates.

        :param x: The image to insert the patch.
        :param patch: The patch to be transformed and inserted.
        :param image_coords: The coordinates of the 4 corners of the transformed, inserted patch of shape
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in pixel units going in clockwise direction, starting with upper
            left corner.
        :return: The input `x` with the patch inserted.
        """
        return insert_transformed_patch(x, patch, image_coords)

    def _check_params(self) -> None:
        super()._check_params()
