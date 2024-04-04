# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
#
# MIT License
#
# Copyright (c) 2022 University of Michigan and University of Wisconsin-Madison
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements helper functions for GRAPHITE attacks.

| Paper link: https://arxiv.org/abs/2002.07088
| Original github link: https://github.com/ryan-feng/GRAPHITE
"""

from typing import Optional, Tuple, Union, TYPE_CHECKING, List
import math
import numpy as np
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
    import torch

_estimator_requirements = (BaseEstimator, ClassifierMixin)


def dist2pixels(dist: float, width: float, obj_width: float = 30) -> float:
    """
    Convert distance to pixels.

    :param dist: Distance to object.
    :param width: Width of image.
    :param obj_width: Width of object.
    :return: Distance in pixels.
    """
    dist_inches = dist * 12
    return 1.0 * dist_inches * width / obj_width


def convert_to_network(x: np.ndarray, net_size: Tuple[int, int], clip_min: float, clip_max: float) -> np.ndarray:
    """
    Convert image to network format.

    :param x: Input image.
    :param net_size: The resolution to resize to in (w, h).
    :param clip_min: Minimum value of an example.
    :param clip_max: Maximum value of an example.
    """
    import cv2

    if net_size is not None:
        x = cv2.resize(x, net_size)
    x = np.clip(x, 0, 1)
    x = x * (clip_max - clip_min) + clip_min
    if len(x.shape) < 3:
        x = x[:, :, np.newaxis]
    return x


def apply_transformation(
    x: np.ndarray,
    mask: np.ndarray,
    pert: np.ndarray,
    angle: float,
    dist: float,
    gamma: float,
    blur: int,
    crop_percent: float,
    crop_off_x: float,
    crop_off_y: float,
    net_size: Tuple[int, int],
    obj_width: float,
    focal: float,
    clip_min: float,
    clip_max: float,
    pts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply transformation to input image.

    :param x: Input image.
    :param mask: Input mask.
    :param pert: Input perturbation.
    :param angle: Angle to rotate image.
    :param dist: Distance in ft for perspective transform.
    :param gamma: Factor for gamma transform.
    :param blur: Kernel width for blurring.
    :param crop_percent: Percent to crop image at.
    :param crop_off_x: x offset for crop.
    :param crop_off_y: y offset for crop.
    :param net_size: Size of the image for the network.
    :param obj_width: Estimated width of object in inches for perspective transform.
    :param focal: Estimated focal length in ft for perspective transform.
    :param clip_min: Minimum value of an example.
    :param clip_max: Maximum value of an example.
    :param pts: A set of points that will set the crop size in the perspective transform.
    :return: Transformed image in network form.
    """
    import cv2

    if blur != 0:
        pert = cv2.GaussianBlur(pert, (blur, blur), 0)

    if len(pert.shape) < 3:
        pert = pert[:, :, np.newaxis]

    att = np.where(mask > 0.5, pert, x)
    att = np.clip(att, 0.0, 1.0)

    dist = dist2pixels(dist, att.shape[1], obj_width)
    focal = dist2pixels(focal, att.shape[1], obj_width)
    att = get_perspective_transform(
        att, angle, att.shape[1], att.shape[0], focal, dist, crop_percent, crop_off_x, crop_off_y, pts
    )

    if len(att.shape) < 3:
        att = att[:, :, np.newaxis]

    # Gamma
    att_uint = (att * 255.0).astype(np.uint8)
    table = np.empty((256), np.uint8)
    for i in range(256):
        table[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    att_uint = cv2.LUT(att_uint, table)  # type: ignore
    att = (att_uint / 255.0).astype(np.float32)
    att = np.clip(att, 0.0, 1.0)

    return convert_to_network(att, net_size, clip_min, clip_max)


def get_transform_params(
    num_xforms: int,
    rotation_range: Tuple[float, float],
    dist_range: Tuple[float, float],
    gamma_range: Tuple[float, float],
    crop_percent_range: Tuple[float, float],
    off_x_range: Tuple[float, float],
    off_y_range: Tuple[float, float],
    blur_kernels: Union[Tuple[int, int], List[int]],
    obj_width: float,
    focal: float,
) -> List[Tuple[float, float, float, int, float, float, float, float, float]]:
    """
    Sample transformation params.

    :param num_xforms: The number of transforms to sample.
    :param rotation_range: The range of the rotation in the perspective transform.
    :param dist_range: The range of the distance (in ft) to be added to the focal length in perspective transform.
    :param gamma_range: The range of the gamma in the gamma transform.
    :param crop_percent_range: The range of the crop percent in the perspective transform.
    :param off_x_range: The range of the x offset (percent) in the perspective transform.
    :param off_y_range: The range of the y offset (percent) in the perspective transform.
    :param blur_kernels: The kernels to blur with.
    :param obj_width: The estimated object width (inches) for perspective transform. 30 by default.
    :param focal: The estimated focal length (ft) for perspective transform. 3 by default.
    :return: List of transforms.
    """
    transforms = []
    for _ in range(num_xforms):
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        dist = np.random.uniform(focal + dist_range[0], focal + dist_range[1])
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        flip_flag = np.random.uniform(0.0, 1.0)
        if int(round(flip_flag)) == 1:
            gamma = 1.0 / gamma

        crop_percent = np.random.uniform(crop_percent_range[0], crop_percent_range[1])
        crop_off_x = np.random.uniform(off_x_range[0], off_x_range[1])
        crop_off_y = np.random.uniform(off_y_range[0], off_y_range[1])
        blur = blur_kernels[int(math.floor(np.random.uniform(0.0, 1.0) * len(blur_kernels)))]
        xform = (angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, obj_width, focal)
        transforms.append(xform)

    return transforms


def add_noise(
    x: np.ndarray,
    mask: np.ndarray,
    lbd: float,
    theta: np.ndarray,
    clip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combines the image and noise to create a perturbed image.

    :param x: Input image.
    :param mask: Mask image.
    :param lbd: lambda multiplier for the perturbation.
    :param theta: The perturbation.
    :param clip: Whether to clip the perturbation.
    :return: image with perturbation, perturbation, mask
    """
    import cv2

    theta_full = cv2.resize(theta, (x.shape[1], x.shape[0])).astype(float)
    if len(theta_full.shape) < 3:
        theta_full = theta_full[:, :, np.newaxis]
    comb = x + lbd * theta_full

    mask_full = cv2.resize(mask, (x.shape[1], x.shape[0])).astype(float)
    if len(mask_full.shape) < 3:
        mask_full = mask_full[:, :, np.newaxis]
    mask_full = np.where(mask_full > 0.5, 1.0, 0.0)

    if clip:
        comb = np.clip(comb, 0, 1)

    pert = np.where(mask_full > 0.5, comb, 0)
    return comb, pert, mask_full


def get_transformed_images(
    x: np.ndarray,
    mask: np.ndarray,
    xforms: List[Tuple[float, float, float, int, float, float, float, float, float]],
    lbd: float,
    theta: np.ndarray,
    net_size: Tuple[int, int],
    clip_min: float,
    clip_max: float,
    pts: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Get transformed images.

    :param x: Input image.
    :param mask: Mask image.
    :param xforms: Transformation parameters.
    :param lbd: lambda multiplier for the perturbation.
    :param theta: The perturbation.
    :param net_size: Size of the image for the network.
    :param clip_min: Minimum value of an example.
    :param clip_max: Maximum value of an example.
    :param pts: A set of points that will set the crop size in the perspective transform.
    :return: List of transformed images.
    """
    att, pert, mask = add_noise(x, mask, lbd, theta, True)

    if len(xforms) == 0:
        return [convert_to_network(att, net_size, clip_min, clip_max)]

    images = []
    for xform in xforms:
        angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, obj_width, focal = xform
        images.append(
            apply_transformation(
                x,
                mask,
                pert,
                angle,
                dist,
                gamma,
                blur,
                crop_percent,
                crop_off_x,
                crop_off_y,
                net_size,
                obj_width,
                focal,
                clip_min,
                clip_max,
                pts,
            )
        )

    return images


def transform_wb(
    x: "torch.Tensor",
    x_adv: "torch.Tensor",
    mask: "torch.Tensor",
    xform: Tuple[float, float, float, int, float, float, float, float, float],
    net_size: Tuple[int, int],
    clip_min: float,
    clip_max: float,
    pts: Optional[np.ndarray],
) -> "torch.Tensor":
    """
    Get transformed image, white-box setting.

    :param x: Original input image.
    :param x_adv: Input image to transform, possibly attacked.
    :param mask: Mask image.
    :param xform: Transformation parameters.
    :param net_size: Size of the image for the network.
    :param clip_min: Minimum value of an example.
    :param clip_max: Maximum value of an example.
    :param pts: A set of points that will set the crop size in the perspective transform.
    :return: Transformed image.
    """
    import torch
    import cv2
    from kornia.enhance import adjust_gamma

    angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, obj_width, focal = xform
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if blur != 0:
        kernel = np.zeros((blur * 2 - 1, blur * 2 - 1))
        kernel[blur - 1, blur - 1] = 1
        kernel = cv2.GaussianBlur(kernel, (blur, blur), 0).astype(float)
        kernel = kernel[blur // 2 : blur // 2 + blur, blur // 2 : blur // 2 + blur]
        kernel = kernel[np.newaxis, :, :]
        kernel = np.repeat(kernel[np.newaxis, :, :, :], x_adv.size()[1], axis=0)
        kernel_torch = torch.from_numpy(kernel)
        blur_torch = torch.nn.Conv2d(
            in_channels=x_adv.size()[1],
            out_channels=x_adv.size()[1],
            kernel_size=blur,
            groups=x_adv.size()[1],
            bias=False,
            padding=blur // 2,
        )
        blur_torch.weight.data = kernel_torch.to(x_adv.dtype)
        blur_torch.weight.requires_grad = False
        blur_torch = blur_torch.to(x_adv.device)
        # the below is done this way to match the black box implementation
        pert = torch.where(mask > 0.5, x_adv, torch.zeros(x_adv.size()).to(mask.device))
        x_adv = torch.where(mask > 0.5, blur_torch(pert), x)
    else:
        x_adv = torch.where(mask > 0.5, x_adv, x)

    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    dist = dist2pixels(dist, x_adv.size()[2], obj_width)
    focal = dist2pixels(focal, x_adv.size()[2], obj_width)
    x_adv = get_perspective_transform_wb(
        x_adv, angle, x_adv.size()[3], x_adv.size()[2], focal, dist, crop_percent, crop_off_x, crop_off_y, pts
    )

    # Gamma
    x_adv = adjust_gamma(x_adv, gamma)

    return convert_to_network_wb(x_adv, net_size, clip_min, clip_max)


def convert_to_network_wb(
    x: "torch.Tensor", net_size: Tuple[int, int], clip_min: float, clip_max: float
) -> "torch.Tensor":
    """
    Convert image to network format.

    :param x: Input image.
    :param net_size: Size of the image for the network.
    :param clip_min: Minimum value of an example.
    :param clip_max: Maximum value of an example.
    :return: Transformed image.
    """
    import torch
    from kornia.geometry.transform import resize

    orig_device = x.device
    x = resize(x, (net_size[1], net_size[0]), align_corners=False)
    x = torch.clamp(x, 0.0, 1.0)
    x = x * (clip_max - clip_min) + clip_min
    return x.to(orig_device)


def get_perspective_transform(
    img: np.ndarray,
    angle: float,
    width: int,
    height: int,
    focal: float,
    dist: float,
    crop_percent: float,
    crop_off_x: float,
    crop_off_y: float,
    pts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes parameters for perspective transform for blackbox attack.

    :param img: Input image.
    :param angle: Angle to rotate.
    :param width: Width of image.
    :param height: Height of image.
    :param focal: Focal length.
    :param dist: Distance for transform.
    :param crop_percent: Percentage for cropping.
    :param crop_off_x: Cropping x offset.
    :param crop_off_y: Cropping y offset.
    :param pts: pts to include in the crop.
    :return: Transformed image.
    """
    import cv2

    perspective_mat, crop_x, crop_y = _get_perspective_transform(
        angle, width, height, focal, dist, crop_percent, crop_off_x, crop_off_y, pts
    )
    dst = cv2.warpPerspective(img, perspective_mat, (crop_x, crop_y), borderMode=cv2.BORDER_REPLICATE)
    return dst


def get_perspective_transform_wb(
    img: "torch.Tensor",
    angle: float,
    width: int,
    height: int,
    focal: float,
    dist: float,
    crop_percent: float,
    crop_off_x: float,
    crop_off_y: float,
    pts: Optional[np.ndarray] = None,
) -> "torch.Tensor":
    """
    Computes perspective transform for whitebox attack.

    :param img: Input image.
    :param angle: Angle to rotate.
    :param width: Width of image.
    :param height: Height of image.
    :param focal: Focal length.
    :param dist: Distance for transform.
    :param crop_percent: Percentage for cropping.
    :param crop_off_x: Cropping x offset.
    :param crop_off_y: Cropping y offset.
    :param pts: pts to include in the crop.
    :return: Transformed image.
    """
    import torch
    from kornia.geometry.transform import warp_perspective

    perspective_mat, crop_x, crop_y = _get_perspective_transform(
        angle, width, height, focal, dist, crop_percent, crop_off_x, crop_off_y, pts
    )
    dst = warp_perspective(
        img,
        torch.from_numpy(perspective_mat).float().to(img.device).unsqueeze(0),
        (crop_y, crop_x),
        align_corners=True,
        padding_mode="border",
    )
    return dst


def _get_perspective_transform(
    angle: float,
    width: int,
    height: int,
    focal: float,
    dist: float,
    crop_percent: float,
    crop_off_x: float,
    crop_off_y: float,
    pts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Computes parameters for perspective transform.

    :param angle: Angle to rotate.
    :param width: Width of image.
    :param height: Height of image.
    :param focal: Focal length.
    :param dist: Distance for transform.
    :param crop_percent: Percentage for cropping.
    :param crop_off_x: Cropping x offset.
    :param crop_off_y: Cropping y offset.
    :param pts: pts to include in the crop.
    :return: perspective transform matrix, crop width, crop_height
    """
    angle = math.radians(angle)
    x_cam_off = width / 2 - math.sin(angle) * dist
    z_cam_off = -math.cos(angle) * dist
    y_cam_off = height / 2

    rot_mat = np.array(
        [
            [math.cos(angle), 0, -math.sin(angle), 0],
            [0, 1, 0, 0],
            [math.sin(angle), 0, math.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )
    c_mat = np.array([[1, 0, 0, -x_cam_off], [0, 1, 0, -y_cam_off], [0, 0, 1, -z_cam_off], [0, 0, 0, 1]])

    rt_mat = np.matmul(rot_mat, c_mat)

    h_mat = np.array(
        [
            [focal * rt_mat[0, 0], focal * rt_mat[0, 1], focal * rt_mat[0, 3]],
            [focal * rt_mat[1, 0], focal * rt_mat[1, 1], focal * rt_mat[1, 3]],
            [rt_mat[2, 0], rt_mat[2, 1], rt_mat[2, 3]],
        ]
    )

    x_off, y_off, crop_size = get_offset_and_crop_size(
        width, height, h_mat, crop_percent, crop_off_x, crop_off_y, focal / dist, pts
    )

    affine_mat = np.array([[1, 0, x_off], [0, 1, y_off], [0, 0, 1]])
    perspective_mat = np.matmul(affine_mat, h_mat)

    if height > width:  # tall and narrow
        crop_x = int(crop_size)
        crop_y = int(round(crop_size / width * height))
    else:  # wide and short or square
        crop_y = int(crop_size)
        crop_x = int(round(crop_size / height * width))

    return perspective_mat, crop_x, crop_y


def get_offset_and_crop_size(
    width: int,
    height: int,
    h_mat: np.ndarray,
    crop_percent: float,
    crop_off_x: float,
    crop_off_y: float,
    ratio: float,
    pts: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Compute offsets and crop size for perspective transform.

    :param w: Width of image.
    :param h: Height of image.
    :param H: Homography matrix.
    :param crop_percent: Percentage for cropping.
    :param crop_off_x: Cropping x offset.
    :param crop_off_y: Cropping y offset.
    :param ratio: Focal length to distance ratio for scaling.
    :param pts: pts to include in the crop.
    :return: x offset, y offset, crop size.
    """
    pts_flag = pts is not None
    if pts is not None:
        pts_copy = [point.copy() for point in pts]
        for i in range(len(pts)):
            pts_copy[i][0, 0] = pts_copy[i][0, 0] * width / pts_copy[i][2, 0]
            pts_copy[i][1, 0] = pts_copy[i][1, 0] * height / pts_copy[i][2, 0]
            pts_copy[i][2, 0] = pts_copy[i][2, 0] * 1.0 / pts_copy[i][2, 0]

    else:
        pts_copy = [
            np.array([[0], [0], [1.0]]),
            np.array([[0], [height], [1.0]]),
            np.array([[width], [0], [1.0]]),
            np.array([[width], [height], [1.0]]),
        ]

    min_x = width
    min_y = height
    max_x = 0
    max_y = 0

    for point in pts_copy:
        new_pt = np.matmul(h_mat, point)
        new_pt /= new_pt[2, 0]

        if new_pt[0, 0] < min_x:
            min_x = new_pt[0, 0]
        if new_pt[0, 0] > max_x:
            max_x = new_pt[0, 0]
        if new_pt[1, 0] < min_y:
            min_y = new_pt[1, 0]
        if new_pt[1, 0] > max_y:
            max_y = new_pt[1, 0]

    if pts_flag:
        if (max_x - min_x) / (max_y - min_y) < width / height:  # result is tall and narrow
            diff_in_size = (max_y - min_y) / height * width - (max_x - min_x)
            orig_size = max_y - min_y if width > height else (max_y - min_y) / height * width
            crop_size = int(round(orig_size * (1.0 - crop_percent)))
            y_off = -min_y - int(round(crop_percent / 2 * orig_size))
            x_off = -min_x + int(round(diff_in_size / 2 - crop_percent / 2 * orig_size))

        else:  # result is wide and short
            diff_in_size = (max_x - min_x) / width * height - (max_y - min_y)
            orig_size = max_x - min_x if height > width else (max_x - min_x) / width * height
            crop_size = int(round(orig_size * (1.0 - crop_percent)))
            x_off = -min_x - int(round(crop_percent / 2 * orig_size))
            y_off = -min_y + int(round(diff_in_size / 2 - crop_percent / 2 * orig_size))

        return x_off + crop_off_x * crop_size, y_off + crop_off_y * crop_size, crop_size

    min_x -= int((width * ratio - (max_x - min_x)) // 2)
    min_y -= int((height * ratio - (max_y - min_y)) // 2)

    crop_size = int(round((1.0 - crop_percent) * min(width, height) * ratio))

    return (
        -min_x - int(round(crop_percent / 2 * width * ratio)),
        -min_y - int(round(crop_percent / 2 * height * ratio)),
        crop_size,
    )


def run_predictions(
    estimator: "CLASSIFIER_NEURALNETWORK_TYPE",
    imgs: List[np.ndarray],
    target: int,
    batch_size: int,
    err_rate: bool = True,
) -> float:
    """
    Run model predictions over batch of input.

    :param estimator: The model.
    :param imgs: Batch of images with different transforms.
    :param target: Target label.
    :param batch_size: Size of batches to process.
    :param err_rate: Whether to return a measure of error or a measure of transform-robutness.
    :return: Either the error rate or transform-robustness depending on err_rate.
    """
    img_tensor = np.zeros((batch_size, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]), dtype=np.float32)
    num_successes = 0
    tar_tensor = np.ones((batch_size)) * target

    count = 0
    for i, img in enumerate(imgs):
        max_y = min(img.shape[0], img_tensor.shape[1])
        max_x = min(img.shape[1], img_tensor.shape[2])
        img_tensor[count, :max_y, :max_x, :] = img[:max_y, :max_x, :]
        count += 1
        if count == batch_size or i == len(imgs) - 1:
            if count < batch_size:
                img_tensor = img_tensor[:count, :, :, :]
                tar_tensor = tar_tensor[:count]

            if estimator.channels_first:
                transposed = np.transpose(img_tensor, (0, 3, 1, 2))
                preds = get_labels_np_array(estimator.predict(transposed, batch_size=batch_size))
            else:
                preds = get_labels_np_array(estimator.predict(img_tensor, batch_size=batch_size))
            np_preds = np.argmax(preds)
            num_successes += (np_preds == tar_tensor).sum().item()
            count = 0

    return_val = (1.0 - num_successes * 1.0 / len(imgs)) if err_rate else num_successes * 1.0 / len(imgs)
    return return_val


def score_fn(mask: np.ndarray, tr_err: float, object_size: float, threshold: float = 0.75, lbd: float = 5):
    """
    Mask scoring function.

    :param mask: The mask.
    :param tr_err: The error in transform-robustness.
    :param object_size: The size of the object.
    :param threshold: The threshold at which to heavily penalize.
    :param lbd: lambda weight multiplier.
    :return: Score of the mask.
    """
    if tr_err > threshold:
        return 10000000

    return lbd * mask.sum() / 3 / (object_size) + tr_err
