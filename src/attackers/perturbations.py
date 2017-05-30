import numpy as np
import numpy.linalg as LA

from src.utils import get_label_conf

def minimal_perturbations(x, model, method_name, sess, eps_step=0.1, max_eps=1., clip_min=None, clip_max=None):
    M = x.shape[0]
    _, pred_y = get_label_conf(model.predict(x))

    min_perturbations = np.zeros(M)
    curr_indexes = np.arange(M)
    eps = eps_step

    while len(curr_indexes) != 0 and eps <= max_eps:
        # get adversarial crafter
        adv_crafter = method_name(model=model, sess=sess)

        # adversarial crafting
        adv_x = adv_crafter.generate_np(x_val=x[curr_indexes], eps=eps, clip_min=clip_min, clip_max=clip_max)
        _, pred_adv_y = get_label_conf(model.predict(adv_x))

        pert_norms = LA.norm(adv_x-x[curr_indexes], axis=(1, 2))

        # update
        assert pert_norms.shape == (len(curr_indexes), 1), pert_norms.shape
        min_perturbations.flat[curr_indexes] = pert_norms

        curr_indexes = np.where(pred_y[curr_indexes] == pred_adv_y)[0]
        eps += eps_step

    return min_perturbations

def empirical_robustness(x, model, method_name, sess, eps_step=0.1, max_eps=1., clip_min=None, clip_max=None):
    """ Computes the Empirical Robustness of a `model` over the sample `x` for a given adversarial crafting method 
    `method_name`, following https://arxiv.org/abs/1511.04599
    
    :param x: 
    :param model: 
    :param method_name: 
    :param sess: 
    :param eps_step: 
    :param max_eps: 
    :param clip_min: 
    :param clip_max: 
    :return: 
    """
    perts = minimal_perturbations(x, model, method_name, sess, eps_step, max_eps, clip_min, clip_max)
    assert perts.shape == (len(x), ), perts.shape
    return np.mean(perts/LA.norm(x))