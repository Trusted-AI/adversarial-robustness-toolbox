from typing import Optional, Tuple
import numpy as np
from art.attacks.evasion.laser_attack.utils import \
    AdvObjectGenerator, DebugInfo, AdversarialObject, ImageGenerator

# https://openaccess.thecvf.com/content/CVPR2021/papers/Duan_Adversarial_Laser_Beam_Effective_Physical-World_Attack_to_DNNs_in_a_CVPR_2021_paper.pdf
# Algorithm 1
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

    params = adv_object_generator.random()
    for _ in range(iterations):
        predicted_class = actual_class
        for sign in [-1, 1]:
            params_prim = adv_object_generator.update_params(params, sign)
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