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
This module implements utility functions for adversarial patch attacks.
"""
import numpy as np


def insert_transformed_patch(x: np.ndarray, patch: np.ndarray, image_coords: np.ndarray):
    """
    Insert patch to image based on given or selected coordinates.

    :param x: A single image of shape HWC to insert the patch.
    :param patch: The patch to be transformed and inserted.
    :param image_coords: The coordinates of the 4 corners of the transformed, inserted patch of shape
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in pixel units going in clockwise direction, starting with upper
            left corner.
    :return: The input `x` with the patch inserted.
    """
    import cv2

    scaling = False

    if np.max(x) <= 1.0:
        scaling = True
        x = (x * 255).astype(np.uint8)
        patch = (patch * 255).astype(np.uint8)

    rows = patch.shape[0]
    cols = patch.shape[1]

    if image_coords.shape[0] == 4:
        patch_coords = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
    else:
        patch_coords = np.array(
            [
                [0, 0],
                [cols - 1, 0],
                [cols - 1, (rows - 1) // 2],
                [cols - 1, rows - 1],
                [0, rows - 1],
                [0, (rows - 1) // 2],
            ]
        )

    # calculate homography
    height, _ = cv2.findHomography(patch_coords, image_coords)

    # warp patch to destination coordinates
    x_out = cv2.warpPerspective(patch, height, (x.shape[1], x.shape[0]), cv2.INTER_CUBIC)

    # mask to aid with insertion
    mask = np.ones(patch.shape)
    mask_out = cv2.warpPerspective(mask, height, (x.shape[1], x.shape[0]), cv2.INTER_CUBIC)

    # save image before adding shadows
    x_neg_patch = np.copy(x)
    x_neg_patch[mask_out != 0] = 0  # negative of the patch space

    if x_neg_patch.shape[2] == 1:
        x_out = np.expand_dims(x_out, axis=2)

    x_out = x_neg_patch.astype("float32") + x_out.astype("float32")

    if scaling:
        x_out = x_out / 255

    return x_out
