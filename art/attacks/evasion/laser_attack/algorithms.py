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
This module implements the greedy search algorithm of the `LaserBeam` attack.

| Paper link: https://arxiv.org/abs/2103.06504
"""
from typing import Optional, Tuple

import numpy as np

from art.attacks.evasion.laser_attack.utils import AdversarialObject, AdvObjectGenerator, DebugInfo, ImageGenerator


def greedy_search(
    image: np.ndarray,
    estimator,
    iterations: int,
    actual_class: int,
    actual_class_confidence: float,
    adv_object_generator: AdvObjectGenerator,
    image_generator: ImageGenerator,
    debug: Optional[DebugInfo] = None,
) -> Tuple[Optional[AdversarialObject], Optional[int]]:
    """
    Greedy search algorithm used to generate parameters of an adversarial object that added to the :image will mislead
    the neural network.
    Based on the paper:
    https://openaccess.thecvf.com/content/CVPR2021/papers/Duan_Adversarial_Laser_Beam_Effective_Physical-World_Attack_to_DNNs_in_a_CVPR_2021_paper.pdf

    :param image: Image to attack.
    :param estimator: Predictor of the image class.
    :param iterations: Maximum number of iterations of the algorithm.
    :param actual_class:
    :param actual_class_confidence:
    :param adv_object_generator: Object responsible for adversarial object generation.
    :param image_generator: Object responsible for image generation.
    :param debug: Optional debug handler.
    """

    params = adv_object_generator.random()
    for _ in range(iterations):
        for sign in [-1, 1]:
            params_prim = adv_object_generator.update_params(params, sign=sign)
            adversarial_image = image_generator.update_image(image, params_prim)
            prediction = estimator.predict(adversarial_image)
            if debug is not None:
                DebugInfo.report(debug, params_prim, np.squeeze(adversarial_image, 0))
            predicted_class = prediction.argmax()
            confidence_adv = prediction[0][actual_class]

            if confidence_adv <= actual_class_confidence:
                params = params_prim
                actual_class_confidence = confidence_adv
                break

        if predicted_class != actual_class:
            return params, predicted_class

    return None, None
