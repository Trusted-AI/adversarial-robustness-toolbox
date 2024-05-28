# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def drop_block2d(input: "torch.tensor", p: float, block_size: int):
    """
    === NOTE ===
    This function is modified from torchvision (torchvision/ops/drop_block.py)
    BSD 3-Clause License
    === ==== ===
    :param input (Tensor[N, C, H, W]): The input tensor or 4-dimensions with the first one
                   being its batch i.e. a batch with ``N`` rows.
    :param p (float): Probability of an element to be dropped.
    :param block_size (int): Size of the block to drop.

    :return: Tensor[N, C, H, W]: The mask of activate pixels.
    """
    import torch

    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}.")
    if input.ndim != 4:
        raise ValueError(f"input should be 4 dimensional. Got {input.ndim} dimensions.")

    N, C, H, W = input.size()
    block_size = min(block_size, W, H)
    # compute the gamma of Bernoulli distribution
    gamma = (p * H * W) / ((block_size ** 2) * ((H - block_size + 1) * (W - block_size + 1)))
    noise = torch.empty((N, 1, H - block_size + 1, W - block_size + 1), dtype=input.dtype, device=input.device)
    noise.bernoulli_(gamma)

    noise = torch.nn.functional.pad(noise, [block_size // 2] * 4, value=0)
    noise = torch.nn.functional.max_pool2d(
        noise, stride=(1, 1), kernel_size=(block_size, block_size), padding=block_size // 2
    )
    mask = 1 - noise
    return mask
