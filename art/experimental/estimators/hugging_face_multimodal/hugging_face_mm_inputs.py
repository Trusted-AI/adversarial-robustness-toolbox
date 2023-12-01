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
This module implements a user-defined dictionary which can support array like functionality to enable compatibility
with ART's tools.
"""
from __future__ import annotations

from collections import UserDict
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch


class HuggingFaceMultiModalInput(UserDict):
    """
    Custom class to allow HF inputs which are in a dictionary to be compatible with ART.
    Allows certain array-like functionality to be performed directly onto the input such as
    some arithmetic operations (addition, subtraction), and python operations
    such as slicing, reshaping, etc to be performed on the correct components of the HF input.
    """

    dtype = object
    shape: Optional[Tuple] = None
    ndim: Optional[int] = None

    def __setitem__(self, key, value):
        """
        Allows setting key-value paris for HuggingFaceMultiModalInput
        :param key: Either a slice, int, or numpy array for array like operations
                    or a string for traditional dictionary setting.
        :param value: Value to set, pixel_values are required for array like operations.
        :return: None
        """
        import torch

        if isinstance(key, slice):
            pixel_values = UserDict.__getitem__(self, "pixel_values")
            original_shape = pixel_values.shape
            with torch.no_grad():
                pixel_values[key] = value["pixel_values"]
                super().__setitem__("pixel_values", pixel_values)
            assert self["pixel_values"].shape == original_shape

        elif isinstance(key, str):
            super().__setitem__(key, value)
            if key == "pixel_values":
                pixel_values = UserDict.__getitem__(self, "pixel_values")
                self.shape = pixel_values.shape
                self.ndim = pixel_values.ndim

        elif isinstance(key, int):
            pixel_values = UserDict.__getitem__(self, "pixel_values")
            original_shape = pixel_values.shape
            with torch.no_grad():
                if isinstance(value, HuggingFaceMultiModalInput):
                    pixel_values[key] = value["pixel_values"]
                else:
                    pixel_values[key] = torch.tensor(value)
                super().__setitem__("pixel_values", pixel_values)
            assert self["pixel_values"].shape == original_shape

        elif isinstance(key, np.ndarray):
            pixel_values = UserDict.__getitem__(self, "pixel_values")
            pixel_values = pixel_values[key]
            super().__setitem__("pixel_values", pixel_values)
            self.shape = pixel_values.shape
            self.ndim = pixel_values.ndim
        else:
            raise ValueError(
                f"Unsupported key {key} with type {type(key)}, "
                f"value {value} for __setitem__ in HuggingFaceMultiModalInput"
            )

    def __getitem__(
        self, item: Union[slice, Tuple, int, str, np.ndarray]
    ) -> Union[HuggingFaceMultiModalInput, "torch.Tensor"]:
        """
        Get value from HuggingFaceMultiModalInput
        :param item: Item to get. If accessing via array like functionality (slice, int, etc) pixel_values are fetched.
                     Else, if passing a string will fetch like a ordinary dictionary
        """
        if isinstance(item, (list, slice, tuple, int, np.ndarray)):
            pixel_values = UserDict.__getitem__(self, "pixel_values")
            pixel_values = pixel_values[item]
            output = HuggingFaceMultiModalInput(**self)
            output["pixel_values"] = pixel_values
            return output
        if item in self.keys():
            return UserDict.__getitem__(self, item)
        raise ValueError("Unsupported item for __getitem__ in HuggingFaceMultiModalInput")

    def __add__(self, other: Union[HuggingFaceMultiModalInput, np.ndarray]) -> HuggingFaceMultiModalInput:
        """
        Performs addition between either two instances of HuggingFaceMultiModalInput on the pixel_values or
        adds a numpy array to the pixel_values if addition is performed between a HuggingFaceMultiModalInput
        and a numpy array.
        :param other: Other value which is being added to self.
        :return: HuggingFaceMultiModalInput with updated "pixel_values".
        """
        import torch

        pixel_values = UserDict.__getitem__(self, "pixel_values")
        dev_id = pixel_values.get_device()

        with torch.no_grad():
            if isinstance(other, HuggingFaceMultiModalInput):
                if dev_id == -1:
                    pixel_values = pixel_values + other["pixel_values"].to("cpu")
                else:
                    pixel_values = pixel_values + other["pixel_values"].to("cuda:" + str(dev_id))
            else:
                if dev_id == -1:
                    pixel_values = pixel_values + torch.tensor(other)
                else:
                    pixel_values = pixel_values + torch.tensor(other).to("cuda:" + str(dev_id))
        output = HuggingFaceMultiModalInput(**self)
        output["pixel_values"] = pixel_values
        return output

    def __sub__(self, other: HuggingFaceMultiModalInput) -> HuggingFaceMultiModalInput:
        """
        Performs subtraction between two instances of HuggingFaceMultiModalInput on the pixel_values.

        :param other: Other value which is being subtracted from self.
        :return: HuggingFaceMultiModalInput with updated "pixel_values".
        """
        if isinstance(other, HuggingFaceMultiModalInput):
            pixel_values = UserDict.__getitem__(self, "pixel_values")
            pixel_values = pixel_values - other["pixel_values"]
            output = HuggingFaceMultiModalInput(**self)
            output["pixel_values"] = pixel_values
        else:
            raise ValueError("Unsupported type for __sub__ in HuggingFaceMultiModalInput")
        return output

    def __mul__(self, other: Union[HuggingFaceMultiModalInput, np.ndarray]) -> HuggingFaceMultiModalInput:
        """
        Performs multiplication between either two instances of HuggingFaceMultiModalInput on the pixel_values or
        adds a numpy array to the pixel_values if addition is performed between a HuggingFaceMultiModalInput
        and a numpy array.
        :param other: Other value which is being multiplied with self.
        :return: HuggingFaceMultiModalInput with updated "pixel_values".
        """
        import torch

        pixel_values = UserDict.__getitem__(self, "pixel_values")

        if isinstance(other, HuggingFaceMultiModalInput):
            pixel_values = pixel_values * other["pixel_values"]
        elif isinstance(other, np.ndarray):
            pixel_values = pixel_values * torch.tensor(other)
        else:
            raise ValueError("Unsupported type for __mul__ in HuggingFaceMultiModalInput")

        output = HuggingFaceMultiModalInput(**self)
        output["pixel_values"] = pixel_values
        return output

    def __len__(self) -> int:
        """
        Fetched the length of the pixel_values
        :return: length of the pixel_values tensor
        """
        pixel_values = UserDict.__getitem__(self, "pixel_values")
        return len(pixel_values)

    def update_pixels(self, pixel_values: torch.Tensor) -> None:
        super().__setitem__("pixel_values", pixel_values)

    def reshape(self, new_shape: Tuple) -> HuggingFaceMultiModalInput:
        """
        Defines reshaping on the HuggingFaceMultiModalInput input.
        :param new_shape: The new shape for the input
        :return: A HuggingFaceMultiModalInput instance with the pixel values having their shape updated.
        """
        import torch

        pixel_values = UserDict.__getitem__(self, "pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        output = HuggingFaceMultiModalInput(**self)
        output["pixel_values"] = torch.reshape(pixel_values, new_shape)
        return output

    def to(self, device: Union["torch.device", str]) -> HuggingFaceMultiModalInput:  # pylint: disable=C0103
        """
        Moves tensors to the supplied device
        :param device: device to move the tensors to.
        :return: A HuggingFaceMultiModalInput instance with tensors moved to the supplied device
        """
        for key in self.keys():
            self[key] = self[key].to(device)
        return self

    @staticmethod
    def is_leaf():
        """
        Enable mypy compatibility
        """
        raise ValueError("Trying to access is_leaf for the whole dictionary. Please use on individual tensors")

    @staticmethod
    def grad():
        """
        Enable mypy compatibility
        """
        raise ValueError("Trying to access is_leaf for the whole dictionary. Please use on individual tensors")
