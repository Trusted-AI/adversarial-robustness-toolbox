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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    import torch
    from art.estimators.certification.derandomized_smoothing.vision_transformers.vit import PyTorchViT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchSmoothedViT:
    """
    Implementation of Certified Patch Robustness via Smoothed Vision Transformers

    | Paper link Accepted version:
        https://openaccess.thecvf.com/content/CVPR2022/papers/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.pdf

    | Paper link Arxiv version (more detail): https://arxiv.org/pdf/2110.07719.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_models(cls, generate_from_null: bool = False) -> List[str]:
        """
        Return the supported model names to the user.

        :param generate_from_null: If to re-check the creation of all the ViTs in timm from scratch.
        :return: A list of compatible models
        """
        import timm
        import torch

        supported_models = [
            "vit_base_patch8_224",
            "vit_base_patch16_18x2_224",
            "vit_base_patch16_224",
            "vit_base_patch16_224_miil",
            "vit_base_patch16_384",
            "vit_base_patch16_clip_224",
            "vit_base_patch16_clip_384",
            "vit_base_patch16_gap_224",
            "vit_base_patch16_plus_240",
            "vit_base_patch16_rpn_224",
            "vit_base_patch16_xp_224",
            "vit_base_patch32_224",
            "vit_base_patch32_384",
            "vit_base_patch32_clip_224",
            "vit_base_patch32_clip_384",
            "vit_base_patch32_clip_448",
            "vit_base_patch32_plus_256",
            "vit_giant_patch14_224",
            "vit_giant_patch14_clip_224",
            "vit_gigantic_patch14_224",
            "vit_gigantic_patch14_clip_224",
            "vit_huge_patch14_224",
            "vit_huge_patch14_clip_224",
            "vit_huge_patch14_clip_336",
            "vit_huge_patch14_xp_224",
            "vit_large_patch14_224",
            "vit_large_patch14_clip_224",
            "vit_large_patch14_clip_336",
            "vit_large_patch14_xp_224",
            "vit_large_patch16_224",
            "vit_large_patch16_384",
            "vit_large_patch32_224",
            "vit_large_patch32_384",
            "vit_medium_patch16_gap_240",
            "vit_medium_patch16_gap_256",
            "vit_medium_patch16_gap_384",
            "vit_small_patch16_18x2_224",
            "vit_small_patch16_36x1_224",
            "vit_small_patch16_224",
            "vit_small_patch16_384",
            "vit_small_patch32_224",
            "vit_small_patch32_384",
            "vit_tiny_patch16_224",
            "vit_tiny_patch16_384",
        ]

        if not generate_from_null:
            return supported_models

        supported = []
        unsupported = []

        models = timm.list_models("vit_*")
        for model in models:
            logger.info("Testing %s creation", model)
            try:
                _ = PyTorchSmoothedViT(
                    model=model,
                    loss=torch.nn.CrossEntropyLoss(),
                    optimizer=torch.optim.SGD,
                    optimizer_params={"lr": 0.01},
                    input_shape=(3, 32, 32),
                    nb_classes=10,
                    ablation_size=4,
                    load_pretrained=False,
                    replace_last_layer=True,
                    verbose=False,
                )
                supported.append(model)
            except (TypeError, AttributeError):
                unsupported.append(model)

        if supported != supported_models:
            logger.warning(
                "Difference between the generated and fixed model list. Although not necessarily "
                "an error, this may point to the timm library being updated."
            )

        return supported

    @staticmethod
    def art_create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> "PyTorchViT":
        """
        Creates a vision transformer using PyTorchViT which controls the forward pass of the model

        :param variant: The name of the vision transformer to load
        :param pretrained: If to load pre-trained weights
        :return: A ViT with the required methods needed for ART
        """

        from timm.models._builder import build_model_with_cfg
        from timm.models.vision_transformer import checkpoint_filter_fn
        from art.estimators.certification.derandomized_smoothing.vision_transformers.vit import PyTorchViT

        return build_model_with_cfg(
            PyTorchViT,
            variant,
            pretrained,
            pretrained_filter_fn=checkpoint_filter_fn,
            **kwargs,
        )
