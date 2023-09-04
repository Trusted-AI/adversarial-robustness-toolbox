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

from art.estimators.certification.derandomized_smoothing.ablators.ablate import BaseAblator


class UpSamplerPyTorch(torch.nn.Module):
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


class ColumnAblatorPyTorch(torch.nn.Module, BaseAblator):
    """
    Pure Pytorch implementation of stripe/column ablation.
    """

    def __init__(
        self,
        ablation_size: int,
        channels_first: bool,
        mode: str,
        to_reshape: bool,
        ablation_mode: str = "column",
        original_shape: Optional[Tuple] = None,
        output_shape: Optional[Tuple] = None,
        algorithm: str = "salman2021",
        device_type: str = "gpu",
    ):
        """
        Creates a column ablator

        :param ablation_size: The size of the column we will retain.
        :param channels_first: If the input is in channels first format. Currently required to be True.
        :param mode: If we are running the algorithm using a CNN or VIT.
        :param to_reshape: If the input requires reshaping.
        :param ablation_mode: The type of ablation to perform.
        :param original_shape: Original shape of the input.
        :param output_shape: Input shape expected by the ViT. Usually means upscaling the input to 224 x 224.
        :param algorithm: Either 'salman2021' or 'levine2020'.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__()

        self.ablation_size = ablation_size
        self.channels_first = channels_first
        self.to_reshape = to_reshape
        self.add_ablation_mask = False
        self.additional_channels = False
        self.algorithm = algorithm
        self.original_shape = original_shape
        self.ablation_mode = ablation_mode

        if self.algorithm == "levine2020":
            self.additional_channels = True
        if self.algorithm == "salman2021" and mode == "ViT":
            self.add_ablation_mask = True

        if device_type == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{cuda_idx}")

        if original_shape is not None and output_shape is not None:
            self.upsample = UpSamplerPyTorch(input_size=original_shape[1], final_size=output_shape[1])

    def ablate(
        self, x: Union[torch.Tensor, np.ndarray], column_pos: int, row_pos: Optional[int] = None
    ) -> torch.Tensor:
        """
        Ablates the input column wise

        :param x: Input data
        :param column_pos: location to start the retained column. NB, if row_ablation_mode is true then this will
                           be used to act on the rows through transposing the image in ColumnAblatorPyTorch.forward
        :param row_pos: Unused.
        :return: The ablated input with 0s where the ablation occurred
        """
        k = self.ablation_size

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if column_pos + k > x.shape[-1]:
            x[:, :, :, (column_pos + k) % x.shape[-1] : column_pos] = 0.0
        else:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k :] = 0.0
        return x

    def forward(
        self, x: Union[torch.Tensor, np.ndarray], column_pos: Optional[int] = None, row_pos=None
    ) -> torch.Tensor:
        """
        Forward pass though the ablator. We insert a new channel to keep track of the ablation location.

        :param x: Input data
        :param column_pos: The start position of the albation
        :param row_pos: Unused.
        :return: The albated input with an extra channel indicating the location of the ablation
        """
        if row_pos is not None:
            raise ValueError("Use column_pos for a ColumnAblator. The row_pos argument is unused")

        if self.original_shape is not None and x.shape[1] != self.original_shape[0] and self.algorithm == "salman2021":
            raise ValueError(f"Ablator expected {self.original_shape[0]} input channels. Recived shape of {x.shape[1]}")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if self.add_ablation_mask:
            ones = torch.torch.ones_like(x[:, 0:1, :, :]).to(self.device)
            x = torch.cat([x, ones], dim=1)

        if self.additional_channels:
            x = torch.cat([x, 1.0 - x], dim=1)

        if self.original_shape is not None and x.shape[1] != self.original_shape[0] and self.additional_channels:
            raise ValueError(
                f"Ablator expected {self.original_shape[0]} input channels. Received shape of {x.shape[1]}"
            )

        if self.ablation_mode == "row":
            x = torch.transpose(x, 3, 2)

        if column_pos is None:
            column_pos = random.randint(0, x.shape[3])

        ablated_x = self.ablate(x, column_pos=column_pos)

        if self.ablation_mode == "row":
            ablated_x = torch.transpose(ablated_x, 3, 2)

        if self.to_reshape:
            ablated_x = self.upsample(ablated_x)
        return ablated_x

    def certify(
        self,
        pred_counts: Union[torch.Tensor, np.ndarray],
        size_to_certify: int,
        label: Union[torch.Tensor, np.ndarray],
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

        # NB! argmax and kthvalue handle ties between predicted counts differently.
        # The original implementation: https://github.com/MadryLab/smoothed-vit/blob/main/src/utils/smoothing.py#L98
        # uses argmax for the model predictions
        # (later called y_smoothed https://github.com/MadryLab/smoothed-vit/blob/main/src/utils/smoothing.py#L230)
        # and kthvalue for the certified predictions.
        # to be consistent with the original implementation we also follow this here.
        top_predicted_class_argmax = torch.argmax(pred_counts, dim=1)

        top_class_counts, top_predicted_class = pred_counts.kthvalue(num_of_classes, dim=1)
        second_class_counts, second_predicted_class = pred_counts.kthvalue(num_of_classes - 1, dim=1)

        cert = (top_class_counts - second_class_counts) > 2 * (size_to_certify + self.ablation_size - 1)

        if self.algorithm == "levine2020":
            tie_break_certs = (
                (top_class_counts - second_class_counts) == 2 * (size_to_certify + self.ablation_size - 1)
            ) & (top_predicted_class < second_predicted_class)
            cert = torch.logical_or(cert, tie_break_certs)

        cert_and_correct = cert & (label == top_predicted_class)

        return cert, cert_and_correct, top_predicted_class_argmax


class BlockAblatorPyTorch(torch.nn.Module, BaseAblator):
    """
    Pure Pytorch implementation of block ablation.
    """

    def __init__(
        self,
        ablation_size: int,
        channels_first: bool,
        mode: str,
        to_reshape: bool,
        original_shape: Optional[Tuple] = None,
        output_shape: Optional[Tuple] = None,
        algorithm: str = "salman2021",
        device_type: str = "gpu",
    ):
        """
        Creates a column ablator

        :param ablation_size: The size of the block we will retain.
        :param channels_first: If the input is in channels first format. Currently required to be True.
        :param mode: If we are running the algorithm using a CNN or VIT.
        :param to_reshape: If the input requires reshaping.
        :param original_shape: Original shape of the input.
        :param output_shape: Input shape expected by the ViT. Usually means upscaling the input to 224 x 224.
        :param algorithm: Either 'salman2021' or 'levine2020'.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__()

        self.ablation_size = ablation_size
        self.channels_first = channels_first
        self.to_reshape = to_reshape
        self.add_ablation_mask = False
        self.additional_channels = False
        self.algorithm = algorithm
        self.original_shape = original_shape

        if self.algorithm == "levine2020":
            self.additional_channels = True
        if self.algorithm == "salman2021" and mode == "ViT":
            self.add_ablation_mask = True

        if device_type == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{cuda_idx}")

        if original_shape is not None and output_shape is not None:
            self.upsample = UpSamplerPyTorch(input_size=original_shape[1], final_size=output_shape[1])

    def ablate(self, x: Union[torch.Tensor, np.ndarray], column_pos: int, row_pos: int) -> torch.Tensor:
        """
        Ablates the input block wise

        :param x: Input data
        :param column_pos: The start position of the albation
        :param row_pos: The row start position of the albation
        :return: The ablated input with 0s where the ablation occurred
        """

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        k = self.ablation_size
        # Column ablations
        if column_pos + k > x.shape[-1]:
            x[:, :, :, (column_pos + k) % x.shape[-1] : column_pos] = 0.0
        else:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k :] = 0.0

        # Row ablations
        if row_pos + k > x.shape[-2]:
            x[:, :, (row_pos + k) % x.shape[-2] : row_pos, :] = 0.0
        else:
            x[:, :, :row_pos, :] = 0.0
            x[:, :, row_pos + k :, :] = 0.0
        return x

    def forward(
        self, x: Union[torch.Tensor, np.ndarray], column_pos: Optional[int] = None, row_pos: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass though the ablator. We insert a new channel to keep track of the ablation location.

        :param x: Input data
        :param column_pos: The start position of the albation
        :return: The albated input with an extra channel indicating the location of the ablation if running in
        """
        if self.original_shape is not None and x.shape[1] != self.original_shape[0] and self.algorithm == "salman2021":
            raise ValueError(f"Ablator expected {self.original_shape[0]} input channels. Recived shape of {x.shape[1]}")

        if column_pos is None:
            column_pos = random.randint(0, x.shape[3])

        if row_pos is None:
            row_pos = random.randint(0, x.shape[2])

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if self.add_ablation_mask:
            ones = torch.torch.ones_like(x[:, 0:1, :, :]).to(self.device)
            x = torch.cat([x, ones], dim=1)

        if self.additional_channels:
            x = torch.cat([x, 1.0 - x], dim=1)

        if self.original_shape is not None and x.shape[1] != self.original_shape[0] and self.additional_channels:
            raise ValueError(f"Ablator expected {self.original_shape[0]} input channels. Recived shape of {x.shape[1]}")

        ablated_x = self.ablate(x, column_pos=column_pos, row_pos=row_pos)

        if self.to_reshape:
            ablated_x = self.upsample(ablated_x)
        return ablated_x

    def certify(
        self,
        pred_counts: Union[torch.Tensor, np.ndarray],
        size_to_certify: int,
        label: Union[torch.Tensor, np.ndarray],
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

        # NB! argmax and kthvalue handle ties between predicted counts differently.
        # The original implementation: https://github.com/MadryLab/smoothed-vit/blob/main/src/utils/smoothing.py#L145
        # uses argmax for the model predictions
        # (later called y_smoothed https://github.com/MadryLab/smoothed-vit/blob/main/src/utils/smoothing.py#L230)
        # and kthvalue for the certified predictions.
        # to be consistent with the original implementation we also follow this here.
        top_predicted_class_argmax = torch.argmax(pred_counts, dim=1)

        num_of_classes = pred_counts.shape[-1]

        top_class_counts, top_predicted_class = pred_counts.kthvalue(num_of_classes, dim=1)
        second_class_counts, second_predicted_class = pred_counts.kthvalue(num_of_classes - 1, dim=1)

        cert = (top_class_counts - second_class_counts) > 2 * (size_to_certify + self.ablation_size - 1) ** 2

        cert_and_correct = cert & (label == top_predicted_class)

        if self.algorithm == "levine2020":
            tie_break_certs = (
                (top_class_counts - second_class_counts) == 2 * (size_to_certify + self.ablation_size - 1) ** 2
            ) & (top_predicted_class < second_predicted_class)
            cert = torch.logical_or(cert, tie_break_certs)
        return cert, cert_and_correct, top_predicted_class_argmax
