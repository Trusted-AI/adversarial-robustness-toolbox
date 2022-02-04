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
This module defines a base class for EoT in PyTorch.
"""
from abc import abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from art.preprocessing.preprocessing import PreprocessorPyTorch

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class EoTPyTorch(PreprocessorPyTorch):
    """
    This module defines a base class for EoT in PyTorch.
    """

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTPyTorch.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

        self.nb_samples = nb_samples
        self.clip_values = clip_values
        EoTPyTorch._check_params(self)

    @abstractmethod
    def _transform(
        self, x: "torch.Tensor", y: Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]], **kwargs
    ) -> Tuple["torch.Tensor", Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]]]:
        """
        Internal method implementing the transformation per input sample.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        raise NotImplementedError

    def forward(
        self, x: "torch.Tensor", y: Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]] = None
    ) -> Tuple["torch.Tensor", Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]]]:
        """
        Apply transformations to inputs `x` and labels `y`.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import torch  # lgtm [py/repeated-import]

        x_preprocess_list = list()
        y_preprocess_list_classification: List[torch.Tensor] = list()
        y_preprocess_list_object_detection: List[List[Dict[str, torch.Tensor]]] = list()

        for i_image in range(x.shape[0]):
            for _ in range(self.nb_samples):
                x_i = x[[i_image]]
                y_i: Optional[Union[torch.Tensor, List[Dict[str, torch.Tensor]]]]
                if y is not None:
                    if isinstance(y, list):
                        y_i = [y[i_image]]
                    else:
                        y_i = y[[i_image]]
                else:
                    y_i = None
                x_preprocess, y_preprocess_i = self._transform(x_i, y_i)
                x_preprocess_list.append(torch.squeeze(x_preprocess, dim=0))

                if y is not None and y_preprocess_i is not None:
                    if isinstance(y_preprocess_i, torch.Tensor):
                        y_preprocess_list_classification.append(torch.squeeze(y_preprocess_i, dim=0))
                    else:
                        y_preprocess_list_object_detection.append(y_preprocess_i)

        x_preprocess = torch.stack(x_preprocess_list, dim=0)
        y_preprocess: Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]]
        if y is None:
            y_preprocess = y
        else:
            if isinstance(y, torch.Tensor):
                y_preprocess = torch.stack(y_preprocess_list_classification, dim=0)
            else:
                y_preprocess = [item for sublist in y_preprocess_list_object_detection for item in sublist]

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:

        if not isinstance(self.nb_samples, int) or self.nb_samples < 1:
            raise ValueError("The number of samples needs to be an integer greater than or equal to 1.")

        if not isinstance(self.clip_values, tuple) or (
            len(self.clip_values) != 2
            or not isinstance(self.clip_values[0], (int, float))
            or not isinstance(self.clip_values[1], (int, float))
            or self.clip_values[0] > self.clip_values[1]
        ):
            raise ValueError("The argument `clip_Values` has to be a float or tuple of two float values as (min, max).")
