import numpy as np
import numpy.linalg as LA

from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod

from src.utils import get_label_conf

def fgsm_minimal_perturbations(x, model, sess, eps_step=0.1, eps_max=1., clip_min=None, clip_max=None):
    M = x.shape[0]
    _, pred_y = get_label_conf(model.predict(x))

    min_perturbations = np.zeros((M, 1))
    curr_indexes = np.arange(M)
    eps = eps_step

    while len(curr_indexes) != 0 and eps <= eps_max:
        # get adversarial crafter
        adv_crafter = FastGradientMethod(model=model, sess=sess)

        # adversarial crafting
        adv_x = adv_crafter.generate_np(x_val=x[curr_indexes], eps=eps, clip_min=clip_min, clip_max=clip_max)
        _, pred_adv_y = get_label_conf(model.predict(adv_x))

        pert_norms = LA.norm(adv_x - x[curr_indexes], axis=(1, 2))

        # update
        min_perturbations.flat[curr_indexes] = pert_norms

        curr_indexes = np.where(pred_y[curr_indexes] == pred_adv_y)[0]
        eps += eps_step

    return min_perturbations

def empirical_robustness(x, model, sess, method_name, method_params):
    """ Computes the Empirical Robustness of a `model` over the sample `x` for a given adversarial crafting method 
    `method_name`, following https://arxiv.org/abs/1511.04599
    
    :param x: 
    :param model: 
    :param method_name: 
    :param sess: 
    :param method_params: 
    :return: 
    """

    if method_name == "fgsm":
        perts = fgsm_minimal_perturbations(x, model, sess, **method_params)
    elif method_name == "jsma":
        jsma = SaliencyMapMethod(model=model, sess=sess)
        adv_x = jsma.generate_np(x, **method_params)
        perts = LA.norm(adv_x-x, axis=(1, 2))

    assert perts.shape == (len(x), 1), perts.shape
    return np.mean(perts/LA.norm(x))