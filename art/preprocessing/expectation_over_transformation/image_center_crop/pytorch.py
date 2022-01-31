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
"""
This module implements Expectation over Transformation preprocessing for image center crop in PyTorch.
"""
import logging
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np

from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class EoTImageCenterCropPyTorch(EoTPyTorch):
    """
    This module implements Expectation over Transformation preprocessing for image center crop in PyTorch.
    """

    params = ["nb_samples", "angles", "clip_values", "label_type"]

    label_types = ["classification", "object_detection"]

    def __init__(
        self,
        nb_samples: int,
        clip_values: "CLIP_VALUES_TYPE",
        size: int = 5,
        label_type: str = "classification",
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTImageCenterCropPyTorch.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
                            for features.
        :param size: Maximal size of the crop on all sides of the image in pixels.
        :param label_type: String defining the type of labels. Currently supported: `classification`, `object_detection`
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit, apply_predict=apply_predict, nb_samples=nb_samples, clip_values=clip_values
        )

        self.size = size
        self.size_range = (0, size)
        self.label_type = label_type
        self._check_params()

    def _transform(
        self, x: "torch.Tensor", y: Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]], **kwargs
    ) -> Tuple["torch.Tensor", Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]]]:
        """
        Center crop an input image and its labels by randomly sampled crop size.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import torch  # lgtm [py/repeated-import]
        import torchvision

        size = np.random.randint(low=self.size_range[0], high=self.size_range[1])

        x_preprocess = torchvision.transforms.functional.resized_crop(
            img=x,
            top=size,
            left=size,
            height=x.shape[-2] - 2 * size,
            width=x.shape[-1] - 2 * size,
            size=x.shape[-2:-1],
            interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
        )

        x_preprocess = torch.clamp(
            input=x_preprocess,
            min=-self.clip_values[0],
            max=self.clip_values[1],
        )

        if self.label_type == "object_detection":

            y_preprocess: List[Dict[str, "torch.Tensor"]] = [{}]

            y_preprocess[0]["boxes"] = torch.clone(y[0]["boxes"])
            y_preprocess[0]["labels"] = torch.clone(y[0]["labels"])

            ratio_h = x.shape[-2] / (x.shape[-2] - 2 * size)
            ratio_w = x.shape[-1] / (x.shape[-1] - 2 * size)

            # top-left corner

            y_preprocess[0]["boxes"][:, 0] -= size
            y_preprocess[0]["boxes"][:, 1] -= size

            y_preprocess[0]["boxes"][:, 0] = y_preprocess[0]["boxes"][:, 0] * ratio_h
            y_preprocess[0]["boxes"][:, 1] = y_preprocess[0]["boxes"][:, 1] * ratio_w

            y_preprocess[0]["boxes"][:, 0] = torch.maximum(torch.tensor(0), y_preprocess[0]["boxes"][:, 0]).int()
            y_preprocess[0]["boxes"][:, 1] = torch.maximum(torch.tensor(0), y_preprocess[0]["boxes"][:, 1]).int()

            # bottom-right corner

            y_preprocess[0]["boxes"][:, 2] -= size
            y_preprocess[0]["boxes"][:, 3] -= size

            y_preprocess[0]["boxes"][:, 2] = y_preprocess[0]["boxes"][:, 2] * ratio_h
            y_preprocess[0]["boxes"][:, 3] = y_preprocess[0]["boxes"][:, 3] * ratio_w

            y_preprocess[0]["boxes"][:, 2] = torch.minimum(
                y_preprocess[0]["boxes"][:, 2], torch.tensor(x.shape[-2])
            ).int()
            y_preprocess[0]["boxes"][:, 3] = torch.minimum(
                y_preprocess[0]["boxes"][:, 3], torch.tensor(x.shape[-1])
            ).int()

        else:

            y_preprocess = y

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:

        if not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("The size be a positive integer.")

        if self.label_type not in self.label_types:
            raise ValueError(
                "The input for label_type needs to be one of {}, currently receiving `{}`.".format(
                    self.label_types, self.label_type
                )
            )
