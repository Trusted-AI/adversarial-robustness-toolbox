# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements resizing for images and object detection bounding boxes in TensorFlow v2.
"""
import logging
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union

from tqdm.auto import tqdm

from art.preprocessing.preprocessing import PreprocessorTensorFlowV2

if TYPE_CHECKING:
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class ImageResizeTensorFlowV2(PreprocessorTensorFlowV2):
    """
    This module implements resizing for images and object detection bounding boxes in TensorFlow v2.
    """

    params = ["height", "width", "channels_first", "label_type", "interpolation", "clip_values", "verbose"]

    label_types = ["classification", "object_detection"]

    def __init__(
        self,
        height: int,
        width: int,
        channels_first: bool = False,
        label_type: str = "classification",
        interpolation: str = "bilinear",
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = True,
        apply_predict: bool = False,
        verbose: bool = False,
    ):
        """
        Create an instance of ImageResizeTensorFlowV2.

        :param height: The height of the resized image.
        :param width: The width of the resized image.
        :param channels_first: Set channels first or last.
        :param label_type: String defining the label type. Currently supported: `classification`, `object_detection`
        :param interpolation: String defining the resizing method. Currently supported: `area`, `bicubic`,
               `bilinear`, `gaussian`, `lanczos3`, `lanczos5`, 'mitchellcubic', `nearest`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.label_type = label_type
        self.interpolation = interpolation
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

    def forward(
        self,
        x: "tf.Tensor",
        y: Optional[Union["tf.Tensor", List[Dict[str, "tf.Tensor"]]]] = None,
    ) -> Tuple["tf.Tensor", Optional[Union["tf.Tensor", List[Dict[str, "tf.Tensor"]]]]]:
        """
        Resize `x` and adjust bounding boxes for labels `y` accordingly.

        :param x: Input samples. A list of samples is also supported.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import tensorflow as tf

        x_preprocess_list = []
        y_preprocess: Optional[Union[tf.Tensor, List[Dict[str, tf.Tensor]]]]
        if y is not None and self.label_type == "object_detection":
            y_preprocess = []
        else:
            y_preprocess = y

        for i, x_i in enumerate(tqdm(x, desc="ImageResizeTensorFlowV2", disable=not self.verbose)):
            if self.channels_first:
                x_i = tf.transpose(x_i, (1, 2, 0))

            # OpenCV swaps height and width
            x_resized = tf.image.resize(x_i, size=(self.height, self.width), method=self.interpolation)

            if self.channels_first:
                x_resized = tf.transpose(x_resized, (2, 0, 1))

            x_preprocess_list.append(x_resized)

            if self.label_type == "object_detection" and y is not None:
                y_resized: Dict[str, tf.Tensor] = {}

                # Copy labels and ensure types
                if isinstance(y, list) and isinstance(y_preprocess, list):
                    y_i = y[i]
                    if isinstance(y_i, dict):
                        y_resized = {k: tf.identity(v) for k, v in y_i.items()}
                    else:
                        raise TypeError("Wrong type for `y` and label_type=object_detection.")
                else:
                    raise TypeError("Wrong type for `y` and label_type=object_detection.")

                # Calculate scaling factor
                height, width, _ = x_i.shape
                height_scale = self.height / height
                width_scale = self.width / width

                # Resize bounding boxes
                boxes = y_resized["boxes"]
                boxes *= tf.constant([width_scale, height_scale, width_scale, height_scale], dtype=boxes.dtype)
                y_resized["boxes"] = boxes

                y_preprocess.append(y_resized)

        x_preprocess = tf.stack(x_preprocess_list)
        if self.clip_values is not None:
            x_preprocess = tf.clip_by_value(x_preprocess, self.clip_values[0], self.clip_values[1])

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:
        if self.height <= 0:
            raise ValueError("The desired image height must be positive.")

        if self.width <= 0:
            raise ValueError("The desired image width must be positive")

        if self.clip_values is not None:
            if len(self.clip_values) != 2:
                raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

            if self.clip_values[0] >= self.clip_values[1]:
                raise ValueError("Invalid `clip_values`: min >= max.")
