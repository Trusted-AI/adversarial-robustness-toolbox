# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2024
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
This module implements the BEYOND detector for adversarial examples detection.

| Paper link: https://openreview.net/pdf?id=S4LqI6CcJ3
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import torch
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE


from art.defences.detector.evasion.evasion_detector import EvasionDetector


class BeyondDetectorPyTorch(EvasionDetector):
    """
      BEYOND detector for adversarial samples detection.
    This detector uses a combination of SSL and target model predictions to detect adversarial examples.

    | Paper link: https://openreview.net/pdf?id=S4LqI6CcJ3
    """

    defence_params = ["target_model", "ssl_model", "augmentations", "aug_num", "alpha", "var_K", "percentile"]

    def __init__(
        self,
        target_classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        ssl_classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        augmentations: Callable,
        aug_num: int = 50,
        alpha: float = 0.8,
        var_K: int = 20,
        percentile: int = 5,
    ) -> None:
        """
        Initialize the BEYOND detector.

        :param target_classifier: The target model to be protected
        :param ssl_classifier: The self-supervised learning model used for feature extraction
        :param augmentations: data augmentations for generating neighborhoods
        :param aug_num: Number of augmentations to apply to each sample (default: 50)
        :param alpha: Weight factor for combining label and representation similarities (default: 0.8)
        :param var_K: Number of top similarities to consider (default: 20)
        :param percentile: using to calculate the threshold
        """
        import torch

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_model = target_classifier.model.to(self.device)
        self.ssl_model = ssl_classifier.model.to(self.device)
        self.aug_num = aug_num
        self.alpha = alpha
        self.var_K = var_K

        self.backbone = self.ssl_model.backbone
        self.model_classifier = self.ssl_model.classifier
        self.projector = self.ssl_model.projector

        self.img_augmentations = augmentations

        self.percentile = percentile  # determine the threshold
        self.threshold: float | None = None

    def _multi_transform(self, img: "torch.Tensor") -> "torch.Tensor":
        import torch

        return torch.stack([self.img_augmentations(img) for _ in range(self.aug_num)], dim=1)

    def _get_metrics(self, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Calculate similarities that combining label consistency and representation similarity for given samples

        :param x: Input samples
        :param batch_size: Batch size for processing
        :return: A report similarities
        """
        import torch
        import torch.nn.functional as F

        samples = torch.from_numpy(x).to(self.device)

        self.target_model.eval()
        self.backbone.eval()
        self.model_classifier.eval()
        self.projector.eval()

        number_batch = int(math.ceil(len(samples) / batch_size))

        similarities_list = []

        with torch.no_grad():
            for index in range(number_batch):
                start = index * batch_size
                end = min((index + 1) * batch_size, len(samples))

                batch_samples = samples[start:end]
                b, c, h, w = batch_samples.shape

                trans_images = self._multi_transform(batch_samples).to(self.device)
                ssl_backbone_out = self.backbone(batch_samples)

                ssl_repre = self.projector(ssl_backbone_out)
                ssl_pred = self.model_classifier(ssl_backbone_out)
                ssl_label = torch.max(ssl_pred, -1)[1]

                aug_backbone_out = self.backbone(trans_images.reshape(-1, c, h, w))
                aug_repre = self.projector(aug_backbone_out)
                aug_pred = self.model_classifier(aug_backbone_out)
                aug_pred = aug_pred.reshape(b, self.aug_num, -1)

                sim_repre = F.cosine_similarity(
                    ssl_repre.unsqueeze(dim=1), aug_repre.reshape(b, self.aug_num, -1), dim=2
                )

                sim_preds = F.cosine_similarity(
                    F.one_hot(ssl_label, num_classes=ssl_pred.shape[-1]).unsqueeze(dim=1),
                    aug_pred,
                    dim=2,
                )

                similarities_list.append(
                    (self.alpha * sim_preds + (1 - self.alpha) * sim_repre).sort(descending=True)[0].cpu().numpy()
                )

        similarities = np.concatenate(similarities_list, axis=0)

        return similarities

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Determine a threshold that covers 95% of clean samples.

        :param x: Clean sample data
        :param y: Clean sample labels (not used in this method)
        :param batch_size: Batch size for processing
        :param nb_epochs: Number of training epochs (not used in this method)
        """
        clean_metrics = self._get_metrics(x=x, batch_size=batch_size)
        k_minus_one_metrics = clean_metrics[:, self.var_K - 1]
        self.threshold = np.percentile(k_minus_one_metrics, q=self.percentile)

    def detect(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
        """
        Detect whether given samples are adversarial

        :param x: Input samples
        :param batch_size: Batch size for processing
        :return: (report, is_adversarial):
            where report containing detection results
            where is_adversarial is a boolean list indicating whether samples are adversarial or not
        """
        if self.threshold is None:
            raise ValueError("Detector has not been fitted. Call fit() before detect().")

        similarities = self._get_metrics(x, batch_size)

        report = similarities[:, self.var_K - 1]
        is_adversarial = report < self.threshold

        return report, is_adversarial
