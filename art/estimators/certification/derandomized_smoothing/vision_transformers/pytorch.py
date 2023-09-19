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

# PatchEmbed class adapted from the implementation in https://github.com/MadryLab/smoothed-vit
#
# Original License:
#
# MIT License
#
# Copyright (c) 2021 Madry Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

"""
Implements functionality for running Vision Transformers in ART
"""
from typing import Optional

import torch
from timm.models.vision_transformer import VisionTransformer


class PatchEmbed(torch.nn.Module):
    """
    Image to Patch Embedding

    Class adapted from the implementation in https://github.com/MadryLab/smoothed-vit

    Original License stated above.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 768):
        """
        Specifies the configuration for the convolutional layer.

        :param patch_size: The patch size used by the ViT.
        :param in_channels: Number of input channels.
        :param embed_dim: The embedding dimension used by the ViT.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj: Optional[torch.nn.Conv2d] = None

    def create(self, patch_size=None, embed_dim=None, device="cpu", **kwargs) -> None:  # pylint: disable=W0613
        """
        Creates a convolution that mimics the embedding layer to be used for the ablation mask to
        track where the image was ablated.

        :param patch_size: The patch size used by the ViT.
        :param embed_dim: The embedding dimension used by the ViT.
        :param device: Which device to set the emdedding layer to.
        :param kwargs: Handles the remaining kwargs from the ViT configuration.
        """

        if patch_size is not None:
            self.patch_size = patch_size
        if embed_dim is not None:
            self.embed_dim = embed_dim

        self.proj = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        w_shape = self.proj.weight.shape
        self.proj.weight = torch.nn.Parameter(torch.ones(w_shape).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedder. We are simply tracking the positions of the ablation mask so no gradients
        are required.

        :param x: Input data corresponding to the ablation mask
        :return: The embedded input
        """
        if self.proj is not None:
            with torch.no_grad():
                x = self.proj(x).flatten(2).transpose(1, 2)
            return x
        raise ValueError("Projection layer not yet created.")


class PyTorchVisionTransformer(VisionTransformer):
    """
    Model-specific class to define the forward pass of the Vision Transformer (ViT) in PyTorch.
    """

    # Make as a class attribute to avoid being included in the
    # state dictionaries of the ViT Model.
    ablation_mask_embedder = PatchEmbed(in_channels=1)

    def __init__(self, **kwargs):
        """
        Create a PyTorchVisionTransformer instance

        :param kwargs: keyword arguments required to create the mask embedder and the vision transformer class
        """
        self.to_drop_tokens = kwargs["drop_tokens"]

        if kwargs["device_type"] == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{cuda_idx}")

        del kwargs["drop_tokens"]
        del kwargs["device_type"]

        super().__init__(**kwargs)
        self.ablation_mask_embedder.create(device=self.device, **kwargs)

        self.in_chans = kwargs["in_chans"]
        self.img_size = kwargs["img_size"]

    @staticmethod
    def drop_tokens(x: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
        """
        Drops the tokens which correspond to fully masked inputs

        :param x: Input data
        :param indexes: positions to be ablated
        :return: Input with tokens dropped where the input was fully ablated.
        """
        x_no_cl, cls_token = x[:, 1:], x[:, 0:1]
        shape = x_no_cl.shape

        # reshape to temporarily remove batch
        x_no_cl = torch.reshape(x_no_cl, shape=(-1, shape[-1]))
        indexes = torch.reshape(indexes, shape=(-1,))
        indexes = indexes.nonzero(as_tuple=True)[0]
        x_no_cl = torch.index_select(x_no_cl, dim=0, index=indexes)
        x_no_cl = torch.reshape(x_no_cl, shape=(shape[0], -1, shape[-1]))
        return torch.cat((cls_token, x_no_cl), dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the ViT.

        :param x: Input data.
        :return: The input processed by the ViT backbone
        """

        ablated_input = False
        if x.shape[1] == self.in_chans + 1:
            ablated_input = True

        if ablated_input:
            x, ablation_mask = x[:, : self.in_chans], x[:, self.in_chans : self.in_chans + 1]

        x = self.patch_embed(x)
        x = self._pos_embed(x)

        if self.to_drop_tokens and ablated_input:
            ones = self.ablation_mask_embedder(ablation_mask)
            to_drop = torch.sum(ones, dim=2)
            indexes = torch.gt(torch.where(to_drop > 1, 1, 0), 0)
            x = self.drop_tokens(x, indexes)

        x = self.norm_pre(x)
        x = self.blocks(x)
        return self.norm(x)
