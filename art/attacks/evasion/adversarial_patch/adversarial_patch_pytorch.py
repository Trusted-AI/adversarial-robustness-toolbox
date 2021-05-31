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
can be printed into the physical world with a common printer. The patch can be used to fool image and video classifiers.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, is_probability, to_categorical

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
        "tensor_board",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        distortion_scale_max: float = 0.0,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        patch_shape: Optional[Tuple[int, int, int]] = None,
        patch_type: str = "circle",
        tensor_board: Union[str, bool] = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.AdversarialPatchPyTorch`.

        :param classifier: A trained classifier.
        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the
               range `[0, 180]`.
        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,
               but less than `scale_max`.
        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but
               larger than `scale_min`.
        :param distortion_scale_max: The maximum distortion scale for perspective transformation in range `[0, 1]`. If
               distortion_scale_max=0.0 the perspective transformation sampling will be disabled.
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape HWC (width, height, nb_channels).
        :param patch_type: The patch type, either circle or square.
        :param tensor_board: Activate summary writer for TensorBoard: Default is `False` and deactivated summary writer.
                             If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory. Provide `path` in type
                             `str` to save in path/CURRENT_DATETIME_HOSTNAME.
                             Use hierarchical folder structure to compare between runs easily. e.g. pass in ‘runs/exp1’,
                             ‘runs/exp2’, etc. for each new experiment to compare across them.
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

        super().__init__(estimator=classifier, tensor_board=tensor_board)
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distortion_scale_max = distortion_scale_max
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        if patch_shape is None:
            self.patch_shape = self.estimator.input_shape
        else:
            self.patch_shape = patch_shape
        self.patch_type = patch_type

        self.image_shape = classifier.input_shape
        self.verbose = verbose
        self._check_params()

        if not self.estimator.channels_first:
            raise ValueError("Input shape has to be wither NCHW or NFCHW.")

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

        if self.patch_shape[1] != self.patch_shape[2]:
            raise ValueError("Patch height and width need to be the same.")

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

        self._optimizer = torch.optim.Adam([self._patch], lr=self.learning_rate)

    def _train_step(
        self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        self._optimizer.zero_grad()
        loss = self._loss(images, target, mask)
        loss.backward(retain_graph=True)
        self._optimizer.step()

        with torch.no_grad():
            self._patch[:] = torch.clamp(
                self._patch, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
            )

        return loss

    def _predictions(self, images: "torch.Tensor", mask: Optional["torch.Tensor"]) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, self._patch, mask=mask)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        predictions = self.estimator._predict_framework(patched_input)  # pylint: disable=W0212

        return predictions

    def _loss(self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"]) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        predictions = self._predictions(images, mask)

        if self.use_logits:
            loss = torch.nn.functional.cross_entropy(
                input=predictions, target=torch.argmax(target, dim=1), reduction="mean"
            )
        else:
            loss = torch.nn.functional.nll_loss(input=predictions, target=torch.argmax(target, dim=1), reduction="mean")

        if not self.targeted:
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
        image_mask = torch.Tensor(np.array(image_mask))
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

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

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
            padding=[pad_h_before, pad_w_before, pad_h_after, pad_w_after],
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
            padding=[pad_h_before, pad_w_before, pad_h_after, pad_w_after],
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
            if scale is None:
                im_scale = np.random.uniform(low=self.scale_min, high=self.scale_max)
            else:
                im_scale = scale

            if mask is None:
                padding_after_scaling_h = (
                    self.image_shape[self.i_h] - im_scale * padded_patch.shape[self.i_h + 1]
                ) / 2.0
                padding_after_scaling_w = (
                    self.image_shape[self.i_w] - im_scale * padded_patch.shape[self.i_w + 1]
                ) / 2.0
                x_shift = np.random.uniform(-padding_after_scaling_w, padding_after_scaling_w)
                y_shift = np.random.uniform(-padding_after_scaling_h, padding_after_scaling_h)
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
        inverted_mask = torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)) - image_mask

        return images * inverted_mask + padded_patch * image_mask

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)

        if y is None:
            logger.info("Setting labels to estimator predictions and running untargeted attack because `y=None`.")
            y = to_categorical(np.argmax(self.estimator.predict(x=x), axis=1), nb_classes=self.estimator.nb_classes)
            self.targeted = False
        else:
            self.targeted = True

        y = check_and_transform_label_format(labels=y, nb_classes=self.estimator.nb_classes)

        # check if logits or probabilities
        y_pred = self.estimator.predict(x=x[[0]])

        if is_probability(y_pred):
            self.use_logits = False
        else:
            self.use_logits = True

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

        for i_iter in trange(self.max_iter, desc="Adversarial Patch PyTorch", disable=not self.verbose):
            if mask is None:
                for images, target in data_loader:
                    _ = self._train_step(images=images, target=target, mask=None)
            else:
                for images, target, mask_i in data_loader:
                    _ = self._train_step(images=images, target=target, mask=mask_i)

            if self.summary_writer is not None:
                self.summary_writer.add_image(
                    "patch",
                    self._patch,
                    global_step=i_iter,
                )

                if hasattr(self.estimator, "compute_losses"):
                    x_patched = self._random_overlay(
                        images=torch.from_numpy(x).to(self.estimator.device), patch=self._patch, mask=mask
                    )
                    losses = self.estimator.compute_losses(x=x_patched, y=torch.from_numpy(y).to(self.estimator.device))

                    for key, value in losses.items():
                        self.summary_writer.add_scalar(
                            "loss/{}".format(key),
                            np.mean(value.detach().cpu().numpy()),
                            global_step=i_iter,
                        )

        return (
            self._patch.detach().cpu().numpy(),
            self._get_circular_patch_mask(nb_samples=1).numpy()[0],
        )

    def _check_mask(self, mask: np.ndarray, x: np.ndarray) -> np.ndarray:
        if mask is not None and (
            (mask.dtype != np.bool)
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
        :param scale: Scale of the applied patch in relation to the classifier input shape.
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
        x = torch.Tensor(x)
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

    def _check_params(self) -> None:
        super()._check_params()

        if not isinstance(self.distortion_scale_max, (float, int)) or 1.0 <= self.distortion_scale_max < 0.0:
            raise ValueError("The maximum distortion scale has to be greater than or equal 0.0 or smaller than 1.0.")

        if self.patch_type not in ["circle", "square"]:
            raise ValueError("The patch type has to be either `circle` or `square`.")
