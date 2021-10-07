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
from art.attacks.evasion.adversarial_patch.utils import insert_transformed_patch
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, is_probability, to_categorical

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class AdversarialTexturePyTorch(EvasionAttack):
    """
    Implementation of the adversarial patch attack for square and rectangular images and videos in PyTorch.

    | Paper link: https://arxiv.org/abs/1712.09665
    """

    attack_params = EvasionAttack.attack_params + [
        "rotation_max",
        "scale_min",
        "scale_max",
        "distortion_scale_max",
        "step_size",
        "max_iter",
        "batch_size",
        "patch_shape",
        "tensor_board",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, )

    def __init__(
        self,
        estimator,
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        distortion_scale_max: float = 0.0,
        step_size: float = 1.0 / 255.0,
        max_iter: int = 500,
        batch_size: int = 16,
        patch_shape: Optional[Tuple[int, int, int]] = None,
        patch_type: str = "circle",
        tensor_board: Union[str, bool] = False,
        verbose: bool = True,
        patch_height=0,
        patch_width=0,
        xmin=0,
        ymin=0,
    ):
        """
        Create an instance of the :class:`.AdversarialTexturePyTorch`.

        :param estimator: A trained estimator.
        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the
               range `[0, 180]`.
        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,
               but less than `scale_max`.
        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but
               larger than `scale_min`.
        :param distortion_scale_max: The maximum distortion scale for perspective transformation in range `[0, 1]`. If
               distortion_scale_max=0.0 the perspective transformation sampling will be disabled.
        :param step_size: The step size.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape HWC (width, height, nb_channels).
        :param patch_type: The patch type, either circle or square.
        :param verbose: Show progress bars.
        """
        import torch  # lgtm [py/repeated-import]
        import torchvision

        # torch_version = list(map(int, torch.__version__.lower().split("+")[0].split(".")))
        # torchvision_version = list(map(int, torchvision.__version__.lower().split("+")[0].split(".")))
        # assert torch_version[0] >= 1 and torch_version[1] >= 7, "AdversarialPatchPyTorch requires torch>=1.7.0"
        # assert (
        #     torchvision_version[0] >= 0 and torchvision_version[1] >= 8
        # ), "AdversarialPatchPyTorch requires torchvision>=0.8.0"

        super().__init__(estimator=estimator, tensor_board=tensor_board)
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distortion_scale_max = distortion_scale_max
        self.step_size = step_size
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.patch_height = patch_height
        self.patch_width = patch_width
        # self.patch_height = int(patch_height / 600 * 224)
        # self.patch_width = int(patch_width / 800 * 224)
        self.xmin = xmin
        self.ymin = ymin
        # self.xmin = int(xmin / 600 * 224)
        # self.ymin = int(ymin / 800 * 224)

        self.image_shape = estimator.input_shape
        self.input_shape = self.estimator.input_shape

        self.nb_dims = len(self.image_shape)

        # if patch_shape is None:
            # if self.nb_dims == 3:
            #     self.patch_shape = self.estimator.input_shape
            # elif self.nb_dims == 4:
            #      self.patch_shape = (self.estimator.input_shape[1], self.estimator.input_shape[2], self.estimator.input_shape[3])
        self.patch_shape = (self.patch_height, self.patch_width, 3)
        # else:
        #     self.patch_shape = patch_shape
        self.patch_type = patch_type

        self.verbose = verbose
        self._check_params()

        # if not self.estimator.channels_first:
        if self.estimator.channels_first:
            raise ValueError("Input shape has to be either NHWC or NFHWC.")

        # self.i_h_patch = 1
        # self.i_w_patch = 2
        self.i_h_patch = 0
        self.i_w_patch = 1

        if self.nb_dims == 3:
            # self.i_h = 1
            # self.i_w = 2
            self.i_h = 0
            self.i_w = 1
        elif self.nb_dims == 4:
            # self.i_h = 2
            # self.i_w = 3
            self.i_h = 1
            self.i_w = 2

        # if self.patch_shape[1] != self.patch_shape[2]:
        # print(self.patch_shape)
        # if self.patch_shape[0] != self.patch_shape[1]:
        #     raise ValueError("Patch height and width need to be the same.")

        if not (self.estimator.postprocessing_defences is None or self.estimator.postprocessing_defences == []):
            raise ValueError(
                "Framework-specific implementation of Adversarial Patch attack does not yet support "
                + "postprocessing defences."
            )

        mean_value = (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2.0 + self.estimator.clip_values[
            0
        ]
        # print(self.patch_shape)
        self._initial_value = np.ones(self.patch_shape) * mean_value
        # self._patch = torch.tensor(self._initial_value, requires_grad=True, device=self.estimator.device)
        self._patch = torch.from_numpy(self._initial_value)
        self._patch.requires_grad = True
        # self._patch.to(self.estimator.device)

        # self._optimizer = torch.optim.SGD([self._patch], lr=1.0)

    def _train_step(
        self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"], y_init, foreground
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        # self.estimator.model.zero_grad()
        loss = self._loss(images, target, mask, y_init, foreground)
        print('loss', loss)
        loss.backward(retain_graph=True)
        # self._optimizer.step()

        # with torch.no_grad():
        # print(self._patch)
        # print(self._patch.grad)
        gradients = self._patch.grad.sign() * self.step_size

        with torch.no_grad():
            self._patch[:] = torch.clamp(
                self._patch + gradients, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
            )
        # print(np.max(self._patch.detach().numpy()))

        return loss

    def _predictions(self, images: "torch.Tensor", mask: Optional["torch.Tensor"], y_init, foreground) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, self._patch, mask=mask, foreground=foreground)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        # predictions = self.estimator._predict_framework(patched_input)  # pylint: disable=W0212
        predictions = self.estimator.predict(patched_input, y_init=y_init)  # pylint: disable=W0212
        # predictions = self.estimator.predict(images, y_init=y_init)  # pylint: disable=W0212

        return predictions

    def _loss(self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"], y_init, foreground) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        y_pred = self._predictions(images, mask, y_init, foreground)

        # print(y_pred)
        # print(target)
        # asdf

        # if self.use_logits:
        #     loss = torch.nn.functional.cross_entropy(
        #         input=predictions, target=torch.argmax(target, dim=1), reduction="mean"
        #     )
        # else:
        #     loss = torch.nn.functional.nll_loss(input=predictions, target=torch.argmax(target, dim=1), reduction="mean")

        # print(type(y_pred[0]["boxes"]))
        # print(type(target["boxes"]))

        loss = torch.nn.L1Loss(size_average=False)(y_pred[0]["boxes"].float(), target["boxes"][0].float())

        return loss

    def _get_circular_patch_mask(self, nb_samples: int, sharpness: int = 40) -> "torch.Tensor":
        """
        Return a circular patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        # print('self.patch_shape', self.patch_shape)
        # print('self.i_h_patch', self.i_h_patch)
        # print('self.i_w_patch', self.i_w_patch)

        # diameter = np.minimum(self.patch_shape[self.i_h_patch], self.patch_shape[self.i_w_patch])
        #
        # if self.patch_type == "circle":
        #     x = np.linspace(-1, 1, diameter)
        #     y = np.linspace(-1, 1, diameter)
        #     x_grid, y_grid = np.meshgrid(x, y, sparse=True)
        #     z_grid = (x_grid ** 2 + y_grid ** 2) ** sharpness
        #     image_mask = 1 - np.clip(z_grid, -1, 1)
        # elif self.patch_type == "square":
        #     # image_mask = np.ones((diameter, diameter))
        image_mask = np.ones((self.patch_height, self.patch_width))

        # image_mask = np.expand_dims(image_mask, axis=0)
        image_mask = np.expand_dims(image_mask, axis=2)
        # print(image_mask.shape, self.patch_shape)
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
        foreground = None
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]
        import torchvision

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        # print('foreground.shape', foreground.shape)

        # smallest_image_edge = np.minimum(self.image_shape[self.i_h], self.image_shape[self.i_w])

        # image_mask = image_mask.permute(0, 3, 1, 2)
        # print(image_mask.shape)

        # image_mask = torchvision.transforms.functional.resize(
        #     img=image_mask,
        #     size=(smallest_image_edge, smallest_image_edge),
        #     interpolation=2,
        # )

        # image_mask = image_mask.permute(0, 2, 3, 1)

        # print(image_mask.shape)
        # asdf

        # pad_h_before = int((self.image_shape[self.i_h] - image_mask.shape[self.i_h_patch + 1]) / 2)
        # pad_h_after = int(self.image_shape[self.i_h] - pad_h_before - image_mask.shape[self.i_h_patch + 1])

        pad_h_before = self.xmin
        # print(self.image_shape[self.i_h], self.xmin)
        # asdf
        # print('image_mask.shape[self.i_h_patch + 1]',  image_mask.shape[self.i_h_patch + 1])
        # print( image_mask.shape)
        # pad_h_after = int(self.image_shape[self.i_h] - pad_h_before - image_mask.shape[self.i_h_patch + 1])
        pad_h_after = int(images.shape[self.i_h+1] - pad_h_before - image_mask.shape[self.i_h_patch + 1])
        # print(pad_h_after)
        # asdf

        # print('images.shape', images.shape)

        # pad_w_before = int((self.image_shape[self.i_w] - image_mask.shape[self.i_w_patch + 1]) / 2)
        # pad_w_after = int(self.image_shape[self.i_w] - pad_w_before - image_mask.shape[self.i_w_patch + 1])

        pad_w_before = self.ymin
        # pad_w_after = int(self.image_shape[self.i_w] - pad_w_before - image_mask.shape[self.i_w_patch + 1])
        pad_w_after = int(images.shape[self.i_w+1] - pad_w_before - image_mask.shape[self.i_w_patch + 1])

        # print(image_mask.shape)

        image_mask = image_mask.permute(0, 3, 1, 2)

        # print(pad_w_before, pad_w_after, pad_h_before, pad_h_after)

        image_mask = torchvision.transforms.functional.pad(
            img=image_mask,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        image_mask = image_mask.permute(0, 2, 3, 1)

        # print(image_mask.shape)

        if self.nb_dims == 4:
            image_mask = torch.unsqueeze(image_mask, dim=1)
            # image_mask = torch.repeat_interleave(image_mask, dim=1, repeats=self.input_shape[0])
            image_mask = torch.repeat_interleave(image_mask, dim=1, repeats=images.shape[1])

        # print(image_mask.shape)
        # asdf

        image_mask = image_mask.float()

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        padded_patch = padded_patch.permute(0, 3, 1, 2)

        # print('padded_patch', padded_patch.shape)

        # padded_patch = torchvision.transforms.functional.resize(
        #     img=padded_patch,
        #     size=(smallest_image_edge, smallest_image_edge),
        #     interpolation=2,
        # )

        padded_patch = torchvision.transforms.functional.pad(
            img=padded_patch,
            # padding=[pad_h_before, pad_w_before, pad_h_after, pad_w_after],
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        padded_patch = padded_patch.permute(0, 2, 3, 1)

        if self.nb_dims == 4:
            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            # padded_patch = torch.repeat_interleave(padded_patch, dim=1, repeats=self.input_shape[0])
            padded_patch = torch.repeat_interleave(padded_patch, dim=1, repeats=images.shape[1])

        # print('padded_patch', padded_patch.shape)
        # sdfg

        padded_patch = padded_patch.float()

        # image_mask_list = list()
        # padded_patch_list = list()
        #
        # for i_sample in range(nb_samples):
        #     if scale is None:
        #         im_scale = np.random.uniform(low=self.scale_min, high=self.scale_max)
        #     else:
        #         im_scale = scale
        #
        #     if mask is None:
        #         padding_after_scaling_h = (
        #             self.image_shape[self.i_h] - im_scale * padded_patch.shape[self.i_h + 1]
        #         ) / 2.0
        #         padding_after_scaling_w = (
        #             self.image_shape[self.i_w] - im_scale * padded_patch.shape[self.i_w + 1]
        #         ) / 2.0
        #         x_shift = np.random.uniform(-padding_after_scaling_w, padding_after_scaling_w)
        #         y_shift = np.random.uniform(-padding_after_scaling_h, padding_after_scaling_h)
        #     else:
        #         mask_2d = mask[i_sample, :, :]
        #
        #         edge_x_0 = int(im_scale * padded_patch.shape[self.i_w + 1]) // 2
        #         edge_x_1 = int(im_scale * padded_patch.shape[self.i_w + 1]) - edge_x_0
        #         edge_y_0 = int(im_scale * padded_patch.shape[self.i_h + 1]) // 2
        #         edge_y_1 = int(im_scale * padded_patch.shape[self.i_h + 1]) - edge_y_0
        #
        #         mask_2d[0:edge_x_0, :] = False
        #         if edge_x_1 > 0:
        #             mask_2d[-edge_x_1:, :] = False
        #         mask_2d[:, 0:edge_y_0] = False
        #         if edge_y_1 > 0:
        #             mask_2d[:, -edge_y_1:] = False
        #
        #         num_pos = np.argwhere(mask_2d).shape[0]
        #         pos_id = np.random.choice(num_pos, size=1)
        #         pos = np.argwhere(mask_2d)[pos_id[0]]
        #         x_shift = pos[1] - self.image_shape[self.i_w] // 2
        #         y_shift = pos[0] - self.image_shape[self.i_h] // 2
        #
        #     phi_rotate = float(np.random.uniform(-self.rotation_max, self.rotation_max))
        #
        #     image_mask_i = image_mask[i_sample]
        #
        #     height = padded_patch.shape[self.i_h + 1]
        #     width = padded_patch.shape[self.i_w + 1]
        #
        #     half_height = height // 2
        #     half_width = width // 2
        #     topleft = [
        #         int(torch.randint(0, int(self.distortion_scale_max * half_width) + 1, size=(1,)).item()),
        #         int(torch.randint(0, int(self.distortion_scale_max * half_height) + 1, size=(1,)).item()),
        #     ]
        #     topright = [
        #         int(torch.randint(width - int(self.distortion_scale_max * half_width) - 1, width, size=(1,)).item()),
        #         int(torch.randint(0, int(self.distortion_scale_max * half_height) + 1, size=(1,)).item()),
        #     ]
        #     botright = [
        #         int(torch.randint(width - int(self.distortion_scale_max * half_width) - 1, width, size=(1,)).item()),
        #         int(torch.randint(height - int(self.distortion_scale_max * half_height) - 1, height, size=(1,)).item()),
        #     ]
        #     botleft = [
        #         int(torch.randint(0, int(self.distortion_scale_max * half_width) + 1, size=(1,)).item()),
        #         int(torch.randint(height - int(self.distortion_scale_max * half_height) - 1, height, size=(1,)).item()),
        #     ]
        #     startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        #     endpoints = [topleft, topright, botright, botleft]
        #
        #     image_mask_i = torchvision.transforms.functional.perspective(
        #         img=image_mask_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
        #     )
        #
        #     image_mask_i = torchvision.transforms.functional.affine(
        #         img=image_mask_i,
        #         angle=phi_rotate,
        #         translate=[x_shift, y_shift],
        #         scale=im_scale,
        #         shear=[0, 0],
        #         resample=0,
        #         fillcolor=None,
        #     )
        #
        #     image_mask_list.append(image_mask_i)
        #
        #     padded_patch_i = padded_patch[i_sample]
        #
        #     padded_patch_i = torchvision.transforms.functional.perspective(
        #         img=padded_patch_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
        #     )
        #
        #     padded_patch_i = torchvision.transforms.functional.affine(
        #         img=padded_patch_i,
        #         angle=phi_rotate,
        #         translate=[x_shift, y_shift],
        #         scale=im_scale,
        #         shear=[0, 0],
        #         resample=0,
        #         fillcolor=None,
        #     )
        #
        #     padded_patch_list.append(padded_patch_i)
        #
        # image_mask = torch.stack(image_mask_list, dim=0)
        # padded_patch = torch.stack(padded_patch_list, dim=0)
        inverted_mask = torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)) - image_mask

        # print("images.shape")
        # print(images.shape)
        # print("inverted_mask.shape")
        # print(inverted_mask.shape)
        # print("padded_patch.shape")
        # print(padded_patch.shape)
        # print("image_mask.shape")
        # print(image_mask.shape)
        # print("foreground.shape")
        # print(foreground.shape)

        # from matplotlib import pyplot as plt
        #
        # fig, axs = plt.subplots(2, 3)
        #
        # idx = 16

        combined = images * inverted_mask \
                   + padded_patch * image_mask \
                   - padded_patch * ~foreground.bool() \
                   + images * ~foreground.bool() * image_mask

        # combined = padded_patch
        # print('combined', combined)

        # grad = images * inverted_mask + images * ~foreground.bool() * image_mask
        # grad = images * ~foreground.bool() * image_mask

        # axs[0, 0].imshow(images.detach().numpy()[0, idx, :, :, :])
        # axs[0, 1].imshow(inverted_mask.detach().numpy()[0, idx, :, :, :])
        # axs[0, 2].imshow(padded_patch.detach().numpy()[0, idx, :, :, :])
        # axs[1, 0].imshow(image_mask.detach().numpy()[0, idx, :, :, :])
        # axs[1, 1].imshow(foreground.detach().numpy()[0, idx, :, :, :])
        # axs[1, 2].imshow(grad.detach().numpy()[0, idx, :, :, :])
        # plt.show()
        #
        # lkj

        return combined

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
        # mask = kwargs.get("mask")
        y_init = kwargs.get("y_init")
        foreground = kwargs.get("foreground")

        # if mask is not None:
        #     mask = mask.copy()
        # mask = self._check_mask(mask=mask, x=x)

        # if y is None:
        #     logger.info("Setting labels to estimator predictions and running untargeted attack because `y=None`.")
        #     y = to_categorical(np.argmax(self.estimator.predict(x=x), axis=1), nb_classes=self.estimator.nb_classes)
        #     self.targeted = False
        # else:
        #     self.targeted = True

        # y = check_and_transform_label_format(labels=y, nb_classes=self.estimator.nb_classes)

        # # check if logits or probabilities
        # y_pred = self.estimator.predict(x=x[[0]])
        #
        # if is_probability(y_pred):
        #     self.use_logits = False
        # else:
        #     self.use_logits = True

        # x_tensor = torch.Tensor(x)
        # y_tensor = torch.Tensor(y)

        class TrackingDataset(torch.utils.data.Dataset):
            def __init__(self, x, y, y_init, foreground):
                # self.x = np.transpose(x, axes=(0, 1, 4, 2, 3))
                # self.x = np.transpose(x, axes=(0, 1, 4, 2, 3))
                self.x = x
                self.y = y
                self.y_init = y_init
                # self.mask = mask
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

        from torch.nn.functional import interpolate

        for i_iter in trange(self.max_iter, desc="Adversarial Texture PyTorch", disable=not self.verbose):
            for images_i, target_i, y_init_i, foreground_i in data_loader:

                # images_i = images_i.permute(0, 1, 4, 2, 3)
                # images_i = interpolate(images_i, size=(3, 224, 224))
                # images_i = images_i.permute(0, 1, 3, 4, 2)

                # images_i = images_i.numpy()
                # from matplotlib import pyplot as plt
                # plt.imshow(images_i[0, 8, :, :, :])
                # plt.show()

                # foreground_i = foreground_i.permute(0, 1, 4, 2, 3)
                # foreground_i = interpolate(foreground_i, size=(3, 224, 224))
                # foreground_i = foreground_i.permute(0, 1, 3, 4, 2)

                _ = self._train_step(images=images_i, target=target_i, mask=None, y_init=y_init_i, foreground=foreground_i)

        # return (
        #     self._patch.detach().cpu().numpy(),
        #     self._get_circular_patch_mask(nb_samples=1).numpy()[0],
        # )
        return self.apply_patch(x=x, scale=1, foreground=foreground)

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
        foreground=None,
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

        # if mask is not None:
        #     mask = mask.copy()
        # mask = self._check_mask(mask=mask, x=x)
        patch = patch_external if patch_external is not None else self._patch
        x = torch.Tensor(x)

        from torch.nn.functional import interpolate

        x_i = x
        # x_i = x_i.permute(0, 1, 4, 2, 3)
        # x_i = interpolate(x_i, size=(3, 224, 224))
        # x_i = x_i.permute(0, 1, 3, 4, 2)

        foreground = torch.Tensor(foreground)

        from torch.nn.functional import interpolate

        foreground_i = foreground
        # foreground_i = foreground_i.permute(0, 1, 4, 2, 3)
        # foreground_i = interpolate(foreground_i, size=(3, 224, 224))
        # foreground_i = foreground_i.permute(0, 1, 3, 4, 2)


        return self._random_overlay(images=x_i, patch=patch, scale=scale, mask=mask, foreground=foreground_i).detach().cpu().numpy()

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

        if self.patch_type not in ["circle", "square"]:
            raise ValueError("The patch type has to be either `circle` or `square`.")
