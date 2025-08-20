# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
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
# WARRANTIES of MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE for any claim, damages or other liability, whether in an action of contract,
# TORT OR OTHERWISE, ARISING from, out of or in connection with the software or the use or other dealings in the
# Software.
"""
This module implements One Pixel Shortcut attacks on Deep Neural Networks.
"""


from __future__ import annotations

import numpy as np

from art.attacks.attack import PoisoningAttackBlackBox


class OnePixelShortcutAttack(PoisoningAttackBlackBox):
    """
    One-Pixel Shortcut (OPS) poisoning attack.
    This attack finds a single pixel (and channel value) that acts as a "shortcut"
    for each class by maximizing a mean-minus-variance objective over that class's
    images. The found pixel coordinate and color are applied to all images of the class
    (labels remain unchanged). Reference: Wu et al. (ICLR 2023).

    | Paper link: https://arxiv.org/abs/2205.12141
    """

    attack_params: list = []  # No external parameters for this attack
    _estimator_requirements: tuple = ()

    def __init__(self):
        super().__init__()

    def _check_params(self):
        # No parameters to validate
        pass

    def poison(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate an OPS-poisoned dataset from clean data.

        :param x: Clean input samples, as a Numpy array of shape (N, H, W, C) or (N, C, H, W), with values in [0, 1].
        :param y: Corresponding labels (shape (N,) or one-hot (N, K)). Required for class-wise perturbation.
        :return: Tuple (x_poisoned, y_poisoned) with one pixel modified per image.
        """
        if y is None:
            raise ValueError("Labels y must be provided for the One-Pixel Shortcut attack.")
        # Copy labels to return (labels are not changed by poisoning)
        y_poison = y.copy()

        # Convert inputs to numpy array (if not already) and determine channel format
        x_array = np.array(x, copy=False)
        if x_array.ndim == 3:
            # Input shape (N, H, W) - single-channel images without explicit channel dim
            x_orig = x_array.reshape((x_array.shape[0], x_array.shape[1], x_array.shape[2], 1)).astype(np.float32)
            channels_first = False
            grayscale = True
        elif x_array.ndim == 4:
            # Determine if format is NCHW or NHWC by examining dimensions
            # Assume channel count is 1, 3, or 4 for common cases (grayscale, RGB, RGBA)
            if x_array.shape[1] in (1, 3, 4) and x_array.shape[-1] not in (1, 3, 4):
                # Likely (N, C, H, W) format
                x_orig = np.transpose(x_array, (0, 2, 3, 1)).astype(np.float32)
                channels_first = True
            elif x_array.shape[-1] in (1, 3, 4) and x_array.shape[1] not in (1, 3, 4):
                # Likely (N, H, W, C) format
                x_orig = x_array.astype(np.float32)
                channels_first = False
            else:
                # Ambiguous case: if both middle and last dims could be channels (e.g. tiny images)
                # Default to treating last dimension as channels if it matches a known channel count
                if x_array.shape[-1] in (1, 3, 4):
                    x_orig = x_array.astype(np.float32)
                    channels_first = False
                else:
                    x_orig = np.transpose(x_array, (0, 2, 3, 1)).astype(np.float32)
                    channels_first = True
            grayscale = x_orig.shape[3] == 1
        else:
            raise ValueError(f"Unsupported input tensor shape: {x_array.shape}")

        # x_orig is now (N, H, W, C) in float32
        n, h, w, c = x_orig.shape
        # Prepare class index labels
        labels = y.copy()
        if labels.ndim > 1:
            labels = labels.argmax(axis=1)
        labels = labels.astype(int)

        # Initialize output poisoned data array
        x_poison = x_orig.copy()

        # Compute optimal pixel for each class
        classes = np.unique(labels)
        for cls in classes:
            idx = np.where(labels == cls)[0]
            if idx.size == 0:
                continue  # skip if no samples for this class
            imgs_c = x_orig[idx]  # subset of images of class `cls`, shape (n_c, H, W, C)
            best_score = -np.inf
            best_coord = None
            best_color = None
            # Determine target color options: extremes (0 or 1 in each channel)
            if c == 1:
                target_options = [
                    np.array([0.0], dtype=x_orig.dtype),
                    np.array([1.0], dtype=x_orig.dtype),
                ]
            else:
                target_options = [np.array(bits, dtype=x_orig.dtype) for bits in np.ndindex(*(2,) * c)]
            # Evaluate each candidate color
            for target_vec in target_options:
                # Compute per-image average difference from target for all pixels
                diffs = np.abs(imgs_c - target_vec)  # shape (n_c, H, W, C)
                per_image_diff = diffs.mean(axis=3)  # shape (n_c, H, W), mean diff per image at each pixel
                # Compute score = mean - var for each pixel position (vectorized over HxW)
                mean_diff_map = per_image_diff.mean(axis=0)  # shape (H, W)
                var_diff_map = per_image_diff.var(axis=0)  # shape (H, W)
                score_map = mean_diff_map - var_diff_map  # shape (H, W)
                # Find the pixel with maximum score for this target
                max_idx_flat = np.argmax(score_map)
                max_score = score_map.ravel()[max_idx_flat]
                if max_score > best_score:
                    best_score = float(max_score)
                    # Convert flat index to 2D coordinates (i, j)
                    best_coord = (max_idx_flat // w, max_idx_flat % w)
                    best_color = target_vec
            # Apply the best pixel perturbation to all images of this class
            if best_coord is not None:
                i_star, j_star = best_coord
                x_poison[idx, i_star, j_star, :] = best_color

        # Restore original data format and type
        if channels_first:
            x_poison = np.transpose(x_poison, (0, 3, 1, 2))
        if grayscale:
            x_poison = x_poison.reshape(n, h, w)
        x_poison = x_poison.astype(x_array.dtype)
        return x_poison, y_poison
