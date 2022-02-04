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
This module implements Expectation over Transformation preprocessing for image rotation in PyTorch.
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


class EoTImageRotationPyTorch(EoTPyTorch):
    """
    This module implements Expectation over Transformation preprocessing for image rotation in PyTorch.
    """

    params = ["nb_samples", "angles", "clip_values", "label_type"]

    label_types = ["classification", "object_detection"]

    def __init__(
        self,
        nb_samples: int,
        clip_values: "CLIP_VALUES_TYPE",
        angles: Union[float, Tuple[float, float], List[float]] = 45.0,
        label_type: str = "classification",
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTImageRotationPyTorch.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
                            for features.
        :param angles: In degrees and counter-clockwise. If a positive scalar it defines the uniform sampling range from
                       negative to positive value. If a tuple of two scalar angles it defines the uniform sampling
                       range from minimum to maximum angles. If a list of scalar values it defines the discrete angles
                       that will be sampled. For `label_type="object_detection"` only a list of multiples of 90 degrees
                       is supported.
        :param label_type: String defining the type of labels. Currently supported: `classification`, `object_detection`
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit, apply_predict=apply_predict, nb_samples=nb_samples, clip_values=clip_values
        )

        self.angles = angles
        self.angles_range = (-angles, angles) if isinstance(angles, (int, float)) else angles
        self.label_type = label_type
        self._check_params()

    def _transform(
        self, x: "torch.Tensor", y: Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]], **kwargs
    ) -> Tuple["torch.Tensor", Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]]]:
        """
        Transformation of an input image and its label by randomly sampled rotation.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import torch  # lgtm [py/repeated-import]
        import torchvision

        if isinstance(self.angles, list):
            angles = np.random.choice(self.angles).item()
        else:
            angles = np.random.uniform(low=self.angles_range[0], high=self.angles_range[1])

        # Ensure channels-first
        channels_first = True
        if x.shape[-1] in [1, 3]:
            x = torch.permute(x, (0, 3, 1, 2))
            channels_first = False

        x_preprocess = torchvision.transforms.functional.rotate(
            img=x, angle=angles, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST, expand=True
        )

        x_preprocess = torch.clamp(
            input=x_preprocess,
            min=-self.clip_values[0],
            max=self.clip_values[1],
        )

        y_preprocess: Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]]

        if self.label_type == "object_detection" and y is not None:

            y_od: List[Dict[str, "torch.Tensor"]] = [{}]

            if isinstance(y, list):
                if isinstance(y[0], dict):
                    y_od[0]["boxes"] = torch.clone(y[0]["boxes"])
                    y_od[0]["labels"] = torch.clone(y[0]["labels"])
                else:
                    raise TypeError("Wrong type for `y` and label_type=object_detection.")
            else:
                raise TypeError("Wrong type for `y` and label_type=object_detection.")

            y_b = y[0]["boxes"]
            image_width = x.shape[2]
            image_height = x.shape[1]
            x_1_arr = y_b[:, 0]
            y_1_arr = y_b[:, 1]
            x_2_arr = y_b[:, 2]
            y_2_arr = y_b[:, 3]
            box_width = x_2_arr - x_1_arr
            box_height = y_2_arr - y_1_arr

            if angles == 0:
                x_1_new = x_1_arr
                y_1_new = y_1_arr
                x_2_new = x_2_arr
                y_2_new = y_2_arr

            elif angles == 90:
                x_1_new = y_1_arr
                y_1_new = image_width - x_1_arr - box_width
                x_2_new = y_1_arr + box_height
                y_2_new = image_width - x_1_arr

            elif angles == 180:
                x_1_new = image_width - x_2_arr
                y_1_new = image_height - y_2_arr
                x_2_new = x_1_new + box_width
                y_2_new = y_1_new + box_height

            elif angles == 270:
                x_1_new = image_height - y_1_arr - box_height
                y_1_new = x_1_arr
                x_2_new = image_height - y_1_arr
                y_2_new = x_1_arr + box_width

            else:
                raise ValueError("The angle is not supported for object detection.")

            y_od[0]["boxes"][:, 0] = x_1_new
            y_od[0]["boxes"][:, 1] = y_1_new
            y_od[0]["boxes"][:, 2] = x_2_new
            y_od[0]["boxes"][:, 3] = y_2_new

            y_preprocess = y_od

        else:

            y_preprocess = y

        if not channels_first:
            x_preprocess = torch.permute(x_preprocess, (0, 2, 3, 1))

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:

        # pylint: disable=R0916
        if (
            self.label_type == "classification"
            and not isinstance(self.angles, (int, float, tuple, list))
            or (
                isinstance(self.angles, tuple)
                and (
                    len(self.angles) != 2
                    or not isinstance(self.angles[0], (int, float))
                    or not isinstance(self.angles[1], (int, float))
                    or self.angles[0] > self.angles[1]
                )
            )
        ):
            raise ValueError("The range of angles must be a float in the range (0.0, 180.0].")

        if self.label_type not in self.label_types:
            raise ValueError(
                "The input for label_type needs to be one of {}, currently receiving `{}`.".format(
                    self.label_types, self.label_type
                )
            )

        if self.label_type == "object_detection":
            if not isinstance(self.angles, list):
                raise ValueError(
                    """For `label_type="object_detection"` only a list of multiples of 90 degrees is supported."""
                )
            for angle in self.angles:
                if divmod(angle, 90)[1] != 0:
                    raise ValueError(
                        """For `label_type="object_detection"` only a list of multiples of 90 degrees is supported."""
                    )
