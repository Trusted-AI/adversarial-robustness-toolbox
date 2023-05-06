"""
This module implements Certified Patch Robustness via Smoothed Vision Transformers

| Paper link Accepted version:
    https://openaccess.thecvf.com/content/CVPR2022/papers/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.pdf

| Paper link Arxiv version (more detail): https://arxiv.org/pdf/2110.07719.pdf
"""
import torch


class UpSampler(torch.nn.Module):
    def __init__(self, input_size, final_size):
        super(UpSampler, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=final_size/input_size)

    def forward(self, x):
        return self.upsample(x)


class ColumnAblator(torch.nn.Module):
    """
    Pure Pytorch implementation of stripe/column ablation.
    """
    def __init__(self, ablation_size: int, channels_first: bool, row_ablation_mode: bool = False):
        super().__init__()
        self.ablation_size = ablation_size
        self.channels_first = channels_first
        self.row_ablation_mode = row_ablation_mode
        self.upsample = UpSampler(input_size=32, final_size=224)

    def ablate(self, x, column_pos):
        k = self.ablation_size
        if column_pos + k > x.shape[-1]:
            x[:, :, :, (column_pos + k) % x.shape[-1]:column_pos] = 0.0
        else:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k:] = 0.0
        return x

    def forward(self, x, column_pos):
        assert x.shape[1] == 3
        ones = torch.torch.ones_like(x[:, 0:1, :, :]).cuda()
        x = torch.cat([x, ones], dim=1)
        x = self.ablate(x, column_pos=column_pos)
        return self.upsample(x)

    def certify(self, predictions, size_to_certify, label):

        num_of_classes = predictions.shape[-1]

        top_class_counts, top_predicted_class = predictions.kthvalue(num_of_classes, dim=1)
        second_class_counts, second_predicted_class = predictions.kthvalue(num_of_classes - 1, dim=1)

        cert = (top_class_counts - second_class_counts) > 2 * (size_to_certify + self.ablation_size - 1)
        cert_and_correct = cert & label == top_predicted_class

        return cert, cert_and_correct
