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
This module implements the adversarial patch attack `AdversarialPatch`. This attack generates an adversarial patch that
can be printed into the physical world with a common printer. The patch can be used to fool image and video estimators.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.adversarial_patch.utils import insert_transformed_patch
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.utils import check_and_transform_label_format, is_probability, to_categorical
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class AdversarialPatchPyTorch(EvasionAttack):
    """
    Implementation of the adversarial patch attack for square and rectangular images and videos in PyTorch.

    | Paper link: https://arxiv.org/abs/1712.09665
    """

    attack_params = EvasionAttack.attack_params + [
        "rotation_max",
        "scale_min",
        "scale_max",
        "distortion_scale_max",
        "learning_rate",
        "max_iter",
        "batch_size",
        "patch_shape",
        "optimizer",
        "targeted",
        "summary_writer",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_NEURALNETWORK_TYPE",
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        distortion_scale_max: float = 0.0,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        patch_shape: Tuple[int, int, int] = (3, 224, 224),
        patch_location: Optional[Tuple[int, int]] = None,
        patch_type: str = "circle",
        optimizer: str = "Adam",
        targeted: bool = True,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.AdversarialPatchPyTorch`.

        :param estimator: A trained estimator.
        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the
               range `[0, 180]`.
        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,
               but less than `scale_max`.
        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but
               larger than `scale_min`.
        :param distortion_scale_max: The maximum distortion scale for perspective transformation in range `[0, 1]`. If
               distortion_scale_max=0.0 the perspective transformation sampling will be disabled.
        :param learning_rate: The learning rate of the optimization. For `optimizer="pgd"` the learning rate gets
                              multiplied with the sign of the loss gradients.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape CHW (nb_channels, height, width).
        :param patch_location: The location of the adversarial patch as a tuple of shape (upper left x, upper left y).
        :param patch_type: The patch type, either circle or square.
        :param optimizer: The optimization algorithm. Supported values: "Adam", and "pgd". "pgd" corresponds to
                          projected gradient descent in L-Inf norm.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        import torch  # lgtm [py/repeated-import]
        import torchvision

        torch_version = list(map(int, torch.__version__.lower().split("+")[0].split(".")))
        torchvision_version = list(map(int, torchvision.__version__.lower().split("+")[0].split(".")))
        assert torch_version[0] >= 1 and torch_version[1] >= 7, "AdversarialPatchPyTorch requires torch>=1.7.0"
        assert (
            torchvision_version[0] >= 0 and torchvision_version[1] >= 8
        ), "AdversarialPatchPyTorch requires torchvision>=0.8.0"

        super().__init__(estimator=estimator, summary_writer=summary_writer)
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distortion_scale_max = distortion_scale_max
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.patch_location = patch_location
        self.patch_type = patch_type

        self.image_shape = estimator.input_shape
        self.targeted = targeted
        self.verbose = verbose
        self._check_params()

        self.i_h_patch = 1
        self.i_w_patch = 2

        self.input_shape = self.estimator.input_shape

        self.nb_dims = len(self.image_shape)
        if self.nb_dims == 3:
            self.i_h = 1
            self.i_w = 2
        elif self.nb_dims == 4:
            self.i_h = 2
            self.i_w = 3

        if self.patch_shape[1] != self.patch_shape[2]:  # pragma: no cover
            raise ValueError("Patch height and width need to be the same.")

        if not (  # pragma: no cover
            self.estimator.postprocessing_defences is None or self.estimator.postprocessing_defences == []
        ):
            raise ValueError(
                "Framework-specific implementation of Adversarial Patch attack does not yet support "
                + "postprocessing defences."
            )

        mean_value = (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2.0 + self.estimator.clip_values[
            0
        ]
        self._initial_value = np.ones(self.patch_shape) * mean_value
        self._patch = torch.tensor(self._initial_value, requires_grad=True, device=self.estimator.device)

        self._optimizer_string = optimizer
        if self._optimizer_string == "Adam":
            self._optimizer = torch.optim.Adam([self._patch], lr=self.learning_rate)

    def _train_step(
        self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        self.estimator.model.zero_grad()
        loss = self._loss(images, target, mask)
        loss.backward(retain_graph=True)

        if self._optimizer_string == "pgd":
            gradients = self._patch.grad.sign() * self.learning_rate

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch + gradients, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
                )
        else:
            self._optimizer.step()

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
                )

        return loss

    def _predictions(
        self, images: "torch.Tensor", mask: Optional["torch.Tensor"], target: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        import torch  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, self._patch, mask=mask)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        predictions, target = self.estimator._predict_framework(patched_input, target)  # pylint: disable=W0212

        return predictions, target

    def _loss(self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"]) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        if isinstance(target, torch.Tensor):

            predictions, target = self._predictions(images, mask, target)

            if self.use_logits:
                loss = torch.nn.functional.cross_entropy(
                    input=predictions, target=torch.argmax(target, dim=1), reduction="mean"
                )
            else:
                loss = torch.nn.functional.nll_loss(
                    input=predictions, target=torch.argmax(target, dim=1), reduction="mean"
                )

        else:
            patched_input = self._random_overlay(images, self._patch, mask=mask)
            patched_input = torch.clamp(
                patched_input,
                min=self.estimator.clip_values[0],
                max=self.estimator.clip_values[1],
            )

            loss = self.estimator.compute_loss(x=patched_input, y=target)

        if (not self.targeted and self._optimizer_string != "pgd") or self.targeted and self._optimizer_string == "pgd":
            loss = -loss

        return loss

    def _get_circular_patch_mask(self, nb_samples: int, sharpness: int = 40) -> "torch.Tensor":
        """
        Return a circular patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        diameter = np.minimum(self.patch_shape[self.i_h_patch], self.patch_shape[self.i_w_patch])

        if self.patch_type == "circle":
            x = np.linspace(-1, 1, diameter)
            y = np.linspace(-1, 1, diameter)
            x_grid, y_grid = np.meshgrid(x, y, sparse=True)
            z_grid = (x_grid ** 2 + y_grid ** 2) ** sharpness
            image_mask = 1 - np.clip(z_grid, -1, 1)
        elif self.patch_type == "square":
            image_mask = np.ones((diameter, diameter))

        image_mask = np.expand_dims(image_mask, axis=0)
        image_mask = np.broadcast_to(image_mask, self.patch_shape)
        image_mask = torch.Tensor(np.array(image_mask)).to(self.estimator.device)
        image_mask = torch.stack([image_mask] * nb_samples, dim=0)
        return image_mask

    def _random_overlay(
        self,
        images: "torch.Tensor",
        patch: "torch.Tensor",
        scale: Optional[float] = None,
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]
        import torchvision

        # Ensure channels-first
        if not self.estimator.channels_first:
            images = torch.permute(images, (0, 3, 1, 2))

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        self.image_shape = images.shape[1:]

        smallest_image_edge = np.minimum(self.image_shape[self.i_h], self.image_shape[self.i_w])

        image_mask = torchvision.transforms.functional.resize(
            img=image_mask,
            size=(smallest_image_edge, smallest_image_edge),
            interpolation=2,
        )

        pad_h_before = int((self.image_shape[self.i_h] - image_mask.shape[self.i_h_patch + 1]) / 2)
        pad_h_after = int(self.image_shape[self.i_h] - pad_h_before - image_mask.shape[self.i_h_patch + 1])

        pad_w_before = int((self.image_shape[self.i_w] - image_mask.shape[self.i_w_patch + 1]) / 2)
        pad_w_after = int(self.image_shape[self.i_w] - pad_w_before - image_mask.shape[self.i_w_patch + 1])

        image_mask = torchvision.transforms.functional.pad(
            img=image_mask,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            image_mask = torch.unsqueeze(image_mask, dim=1)
            image_mask = torch.repeat_interleave(image_mask, dim=1, repeats=self.input_shape[0])

        image_mask = image_mask.float()

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        padded_patch = torchvision.transforms.functional.resize(
            img=padded_patch,
            size=(smallest_image_edge, smallest_image_edge),
            interpolation=2,
        )

        padded_patch = torchvision.transforms.functional.pad(
            img=padded_patch,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            padded_patch = torch.repeat_interleave(padded_patch, dim=1, repeats=self.input_shape[0])

        padded_patch = padded_patch.float()

        image_mask_list = list()
        padded_patch_list = list()

        for i_sample in range(nb_samples):
            if self.patch_location is None:
                if scale is None:
                    im_scale = np.random.uniform(low=self.scale_min, high=self.scale_max)
                else:
                    im_scale = scale
            else:
                im_scale = self.patch_shape[self.i_h] / smallest_image_edge

            if mask is None:
                if self.patch_location is None:
                    padding_after_scaling_h = (
                        self.image_shape[self.i_h] - im_scale * padded_patch.shape[self.i_h + 1]
                    ) / 2.0
                    padding_after_scaling_w = (
                        self.image_shape[self.i_w] - im_scale * padded_patch.shape[self.i_w + 1]
                    ) / 2.0
                    x_shift = np.random.uniform(-padding_after_scaling_w, padding_after_scaling_w)
                    y_shift = np.random.uniform(-padding_after_scaling_h, padding_after_scaling_h)
                else:
                    padding_h = int(math.floor(self.image_shape[self.i_h] - self.patch_shape[self.i_h]) / 2.0)
                    padding_w = int(math.floor(self.image_shape[self.i_w] - self.patch_shape[self.i_w]) / 2.0)
                    x_shift = -padding_w + self.patch_location[0]
                    y_shift = -padding_h + self.patch_location[1]
            else:
                mask_2d = mask[i_sample, :, :]

                edge_x_0 = int(im_scale * padded_patch.shape[self.i_w + 1]) // 2
                edge_x_1 = int(im_scale * padded_patch.shape[self.i_w + 1]) - edge_x_0
                edge_y_0 = int(im_scale * padded_patch.shape[self.i_h + 1]) // 2
                edge_y_1 = int(im_scale * padded_patch.shape[self.i_h + 1]) - edge_y_0

                mask_2d[0:edge_x_0, :] = False
                if edge_x_1 > 0:
                    mask_2d[-edge_x_1:, :] = False
                mask_2d[:, 0:edge_y_0] = False
                if edge_y_1 > 0:
                    mask_2d[:, -edge_y_1:] = False

                num_pos = np.argwhere(mask_2d).shape[0]
                pos_id = np.random.choice(num_pos, size=1)
                pos = np.argwhere(mask_2d)[pos_id[0]]
                x_shift = pos[1] - self.image_shape[self.i_w] // 2
                y_shift = pos[0] - self.image_shape[self.i_h] // 2

            phi_rotate = float(np.random.uniform(-self.rotation_max, self.rotation_max))

            image_mask_i = image_mask[i_sample]

            height = padded_patch.shape[self.i_h + 1]
            width = padded_patch.shape[self.i_w + 1]

            half_height = height // 2
            half_width = width // 2
            topleft = [
                int(torch.randint(0, int(self.distortion_scale_max * half_width) + 1, size=(1,)).item()),
                int(torch.randint(0, int(self.distortion_scale_max * half_height) + 1, size=(1,)).item()),
            ]
            topright = [
                int(torch.randint(width - int(self.distortion_scale_max * half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(0, int(self.distortion_scale_max * half_height) + 1, size=(1,)).item()),
            ]
            botright = [
                int(torch.randint(width - int(self.distortion_scale_max * half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(height - int(self.distortion_scale_max * half_height) - 1, height, size=(1,)).item()),
            ]
            botleft = [
                int(torch.randint(0, int(self.distortion_scale_max * half_width) + 1, size=(1,)).item()),
                int(torch.randint(height - int(self.distortion_scale_max * half_height) - 1, height, size=(1,)).item()),
            ]
            startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            endpoints = [topleft, topright, botright, botleft]

            image_mask_i = torchvision.transforms.functional.perspective(
                img=image_mask_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
            )

            image_mask_i = torchvision.transforms.functional.affine(
                img=image_mask_i,
                angle=phi_rotate,
                translate=[x_shift, y_shift],
                scale=im_scale,
                shear=[0, 0],
                resample=0,
                fillcolor=None,
            )

            image_mask_list.append(image_mask_i)

            padded_patch_i = padded_patch[i_sample]

            padded_patch_i = torchvision.transforms.functional.perspective(
                img=padded_patch_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
            )

            padded_patch_i = torchvision.transforms.functional.affine(
                img=padded_patch_i,
                angle=phi_rotate,
                translate=[x_shift, y_shift],
                scale=im_scale,
                shear=[0, 0],
                resample=0,
                fillcolor=None,
            )

            padded_patch_list.append(padded_patch_i)

        image_mask = torch.stack(image_mask_list, dim=0)
        padded_patch = torch.stack(padded_patch_list, dim=0)
        inverted_mask = (
            torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)).to(self.estimator.device) - image_mask
        )

        patched_images = images * inverted_mask + padded_patch * image_mask

        if not self.estimator.channels_first:
            patched_images = torch.permute(patched_images, (0, 2, 3, 1))

        return patched_images

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: An array with the original input images of shape NCHW or input videos of shape NFCHW.
        :param y: An array with the original true labels.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :return: An array with adversarial patch and an array of the patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        shuffle = kwargs.get("shuffle", True)
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)

        if self.patch_location is not None and mask is not None:
            raise ValueError("Masks can only be used if the `patch_location` is `None`.")

        if y is None:  # pragma: no cover
            logger.info("Setting labels to estimator predictions and running untargeted attack because `y=None`.")
            y = to_categorical(np.argmax(self.estimator.predict(x=x), axis=1), nb_classes=self.estimator.nb_classes)

        if hasattr(self.estimator, "nb_classes"):
            y = check_and_transform_label_format(labels=y, nb_classes=self.estimator.nb_classes)

            # check if logits or probabilities
            y_pred = self.estimator.predict(x=x[[0]])

            if is_probability(y_pred):
                self.use_logits = False
            else:
                self.use_logits = True

        if isinstance(y, np.ndarray):
            x_tensor = torch.Tensor(x)
            y_tensor = torch.Tensor(y)

            if mask is None:
                dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    drop_last=False,
                )
            else:
                mask_tensor = torch.Tensor(mask)
                dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, mask_tensor)
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    drop_last=False,
                )
        else:

            class ObjectDetectionDataset(torch.utils.data.Dataset):
                """
                Object detection dataset in PyTorch.
                """

                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def __len__(self):
                    return self.x.shape[0]

                def __getitem__(self, idx):
                    img = torch.from_numpy(self.x[idx])

                    target = {}
                    target["boxes"] = torch.from_numpy(self.y[idx]["boxes"])
                    target["labels"] = torch.from_numpy(self.y[idx]["labels"])
                    target["scores"] = torch.from_numpy(self.y[idx]["scores"])

                    return img, target

            class ObjectDetectionDatasetMask(torch.utils.data.Dataset):
                """
                Object detection dataset in PyTorch.
                """

                def __init__(self, x, y, mask):
                    self.x = x
                    self.y = y
                    self.mask = mask

                def __len__(self):
                    return self.x.shape[0]

                def __getitem__(self, idx):
                    img = torch.from_numpy(self.x[idx])

                    target = {}
                    target["boxes"] = torch.from_numpy(y[idx]["boxes"])
                    target["labels"] = torch.from_numpy(y[idx]["labels"])
                    target["scores"] = torch.from_numpy(y[idx]["scores"])
                    mask_i = torch.from_numpy(self.mask[idx])

                    return img, target, mask_i

            dataset_object_detection: Union[ObjectDetectionDataset, ObjectDetectionDatasetMask]
            if mask is None:
                dataset_object_detection = ObjectDetectionDataset(x, y)
            else:
                dataset_object_detection = ObjectDetectionDatasetMask(x, y, mask)

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset_object_detection,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=False,
            )

        for i_iter in trange(self.max_iter, desc="Adversarial Patch PyTorch", disable=not self.verbose):
            if mask is None:
                for images, target in data_loader:
                    images = images.to(self.estimator.device)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.estimator.device)
                    else:
                        target["boxes"] = target["boxes"].to(self.estimator.device)
                        target["labels"] = target["labels"].to(self.estimator.device)
                        target["scores"] = target["scores"].to(self.estimator.device)
                    _ = self._train_step(images=images, target=target, mask=None)
            else:
                for images, target, mask_i in data_loader:
                    images = images.to(self.estimator.device)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.estimator.device)
                    else:
                        target["boxes"] = target["boxes"].to(self.estimator.device)
                        target["labels"] = target["labels"].to(self.estimator.device)
                        target["scores"] = target["scores"].to(self.estimator.device)
                    mask_i = mask_i.to(self.estimator.device)
                    _ = self._train_step(images=images, target=target, mask=mask_i)

            # Write summary
            if self.summary_writer is not None:  # pragma: no cover
                x_patched = (
                    self._random_overlay(
                        images=torch.from_numpy(x).to(self.estimator.device), patch=self._patch, mask=mask
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_iter,
                    grad=None,
                    patch=self._patch,
                    estimator=self.estimator,
                    x=x_patched,
                    y=y,
                    targeted=self.targeted,
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return (
            self._patch.detach().cpu().numpy(),
            self._get_circular_patch_mask(nb_samples=1).cpu().numpy()[0],
        )

    def _check_mask(self, mask: np.ndarray, x: np.ndarray) -> np.ndarray:
        if mask is not None and (  # pragma: no cover
            (mask.dtype != bool)
            or not (mask.shape[0] == 1 or mask.shape[0] == x.shape[0])
            or not (mask.shape[1] == x.shape[self.i_h + 1] and mask.shape[2] == x.shape[self.i_w + 1])
        ):
            raise ValueError(
                "The shape of `mask` has to be equal to the shape of a single samples (1, H, W) or the"
                "shape of `x` (N, H, W) without their channel dimensions."
            )

        if mask is not None and mask.shape[0] == 1:
            mask = np.repeat(mask, repeats=x.shape[0], axis=0)

        return mask

    def apply_patch(
        self,
        x: np.ndarray,
        scale: float,
        patch_external: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        A function to apply the learned adversarial patch to images or videos.

        :param x: Instances to apply randomly transformed patch.
        :param scale: Scale of the applied patch in relation to the estimator input shape.
        :param patch_external: External patch to apply to images `x`.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched samples.
        """
        import torch  # lgtm [py/repeated-import]

        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)
        patch = patch_external if patch_external is not None else self._patch
        x = torch.Tensor(x).to(self.estimator.device)
        return self._random_overlay(images=x, patch=patch, scale=scale, mask=mask).detach().cpu().numpy()

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

        if not isinstance(self.distortion_scale_max, (float, int)) or 1.0 <= self.distortion_scale_max < 0.0:
            raise ValueError("The maximum distortion scale has to be greater than or equal 0.0 or smaller than 1.0.")

        if self.patch_location is not None and not (
            isinstance(self.patch_location, tuple)
            and len(self.patch_location) == 2
            and isinstance(self.patch_location[0], int)
            and self.patch_location[0] >= 0
            and isinstance(self.patch_location[1], int)
            and self.patch_location[1] >= 0
        ):
            raise ValueError(
                "The patch location has to be either `None` or a tuple of two integers greater than or equal 0."
            )

        if self.patch_type not in ["circle", "square"]:
            raise ValueError("The patch type has to be either `circle` or `square`.")
