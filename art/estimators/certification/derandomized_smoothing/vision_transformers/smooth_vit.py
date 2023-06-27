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
This module implements Certified Patch Robustness via Smoothed Vision Transformers

| Paper link Accepted version:
    https://openaccess.thecvf.com/content/CVPR2022/papers/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.pdf

| Paper link Arxiv version (more detail): https://arxiv.org/pdf/2110.07719.pdf
"""

from typing import Optional, Union, Tuple
import random

import numpy as np
import torch


class UpSampler(torch.nn.Module):
    """
    Resizes datasets to the specified size.
    Usually for upscaling datasets like CIFAR to Imagenet format
    """

    def __init__(self, input_size: int, final_size: int) -> None:
        """
        Creates an upsampler to make the supplied data match the pre-trained ViT format

        :param input_size: Size of the current input data
        :param final_size: Desired final size
        """
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=final_size / input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the upsampler.

        :param x: Input data
        :return: The upsampled input data
        """
        return self.upsample(x)


class ColumnAblator(torch.nn.Module):
    """
    Pure Pytorch implementation of stripe/column ablation.
    """

    def __init__(
        self,
        ablation_size: int,
        channels_first: bool,
        to_reshape: bool = False,
        original_shape: Optional[Tuple] = None,
        output_shape: Optional[Tuple] = None,
        device_type: str = "gpu",
    ):
        """
        Creates a column ablator

        :param ablation_size: The size of the column we will retain.
        :param channels_first: If the input is in channels first format. Currently required to be True.
        :param to_reshape: If the input requires reshaping.
        :param original_shape: Original shape of the input.
        :param output_shape: Input shape expected by the ViT. Usually means upscaling the input to 224 x 224.
        """
        super().__init__()
        self.ablation_size = ablation_size
        self.channels_first = channels_first
        self.to_reshape = to_reshape

        if device_type == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{cuda_idx}")

        if original_shape is not None and output_shape is not None:
            self.upsample = UpSampler(input_size=original_shape[1], final_size=output_shape[1])

    def ablate(self, x: torch.Tensor, column_pos: int) -> torch.Tensor:
        """
        Ablates the input colum wise

        :param x: Input data
        :param column_pos: The start position of the albation
        :return: The ablated input with 0s where the ablation occurred
        """
        k = self.ablation_size
        if column_pos + k > x.shape[-1]:
            x[:, :, :, (column_pos + k) % x.shape[-1] : column_pos] = 0.0
        else:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k :] = 0.0
        return x

    def forward(self, x: Union[torch.Tensor, np.ndarray], column_pos: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass though the ablator. We insert a new channel to keep track of the ablation location.

        :param x: Input data
        :param column_pos: The start position of the albation
        :return: The albated input with an extra channel indicating the location of the ablation
        """
        assert x.shape[1] == 3

        if column_pos is None:
            column_pos = random.randint(0, x.shape[3])

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        ones = torch.torch.ones_like(x[:, 0:1, :, :]).to(self.device)
        x = torch.cat([x, ones], dim=1)
        x = self.ablate(x, column_pos=column_pos)
        if self.to_reshape:
            x = self.upsample(x)
        return x

    def certify(
        self, pred_counts: Union[torch.Tensor, np.ndarray], size_to_certify: int, label: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs certification of the predictions

        :param pred_counts: The model predictions over the ablated data.
        :param size_to_certify: The patch size we wish to check certification against
        :param label: The ground truth labels
        :return: A tuple consisting of: the certified predictions,
                 the predictions which were certified and also correct,
                 and the most predicted class across the different ablations on the input.
        """
        if isinstance(pred_counts, np.ndarray):
            pred_counts = torch.from_numpy(pred_counts).to(self.device)

        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).to(self.device)

        num_of_classes = pred_counts.shape[-1]

        top_class_counts, top_predicted_class = pred_counts.kthvalue(num_of_classes, dim=1)
        second_class_counts, _ = pred_counts.kthvalue(num_of_classes - 1, dim=1)

        cert = (top_class_counts - second_class_counts) > 2 * (size_to_certify + self.ablation_size - 1)

        cert_and_correct = cert & (label == top_predicted_class)

        return cert, cert_and_correct, top_predicted_class
