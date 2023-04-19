
"""
This module implements Certified Patch Robustness via Smoothed Vision Transformers

| Paper link: https://openaccess.thecvf.com/content/CVPR2022/papers/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.pdf
"""
import torch as nn

class ColumnAblator(nn.Module):
    """
    Pure Pytorch implementation of stripe/column ablation.
    """
    def __init__(self, ablation_size: int, channels_first: bool, row_ablation_mode: bool = False):
        super().__init__()
        self.ablation_size = ablation_size
        self.channels_first = channels_first
        self.row_ablation_mode = row_ablation_mode

    def forward(self):
        raise NotImplementedError
