import config

import numpy as np
import numpy.linalg as LA

from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod, BasicIterativeMethod

from src.utils import get_label_conf

supported_methods = { "fgsm": {"class": FastGradientMethod,
                               "params": {"eps_step": 0.1, "clip_min": 0., "clip_max": 1.}},
                      "jsma": {"class": SaliencyMapMethod,
                               "params": {"theta": 1., "gamma": 0.01, "clip_min": 0., "clip_max": 1.}}
                      }

def get_crafter(method, model, session, params=None):

    try:
        crafter = supported_methods[method]["class"](model, sess=session)
    except:
        raise NotImplementedError("{} crafting method not supported.".format(method))

    if params:
        crafter.parse_params(**params)
    else:
        crafter.parse_params(**supported_methods[method]["params"])

    return crafter

def fgsm_minimal_perturbations(x, model, sess, eps_step=0.1, eps_max=1., params=None):
    M = x.shape[0]
    _, pred_y = get_label_conf(model.predict(x))

    min_perturbations = np.zeros(x.shape)
    curr_indexes = np.arange(M)
    eps = eps_step

    while len(curr_indexes) != 0 and eps <= eps_max:
        # get adversarial crafter
        adv_crafter = get_crafter("fgsm", model, sess, params)

        # adversarial crafting
        adv_x = adv_crafter.generate_np(x_val=x[curr_indexes], eps=eps)
        _, pred_adv_y = get_label_conf(model.predict(adv_x))

        # update
        min_perturbations[curr_indexes] = adv_x - x[curr_indexes]

        curr_indexes = np.where(pred_y[curr_indexes] == pred_adv_y)[0]
        eps += eps_step

    return min_perturbations

# def bim_minimal_perturbations(x, model, sess, eps_step=0.1, eps_max=1., clip_min=None, clip_max=None):
#     M = x.shape[0]
#     _, pred_y = get_label_conf(model.predict(x))
#
#     min_perturbations = np.zeros((M, 1))
#     curr_indexes = np.arange(M)
#     eps = eps_step
#
#     while len(curr_indexes) != 0 and eps <= eps_max:
#         # get adversarial crafter
#         adv_crafter = BasicIterativeMethod(model=model, sess=sess)
#
#         # adversarial crafting
#         adv_x = adv_crafter.generate_np(x_val=x[curr_indexes], eps=eps, clip_min=clip_min, clip_max=clip_max)
#         _, pred_adv_y = get_label_conf(model.predict(adv_x))
#
#         pert_norms = LA.norm(adv_x - x[curr_indexes], axis=(1, 2))
#
#         # update
#         min_perturbations.flat[curr_indexes] = pert_norms
#
#         curr_indexes = np.where(pred_y[curr_indexes] == pred_adv_y)[0]
#         eps += eps_step
#
#     return min_perturbations

def empirical_robustness(x, model, sess, method_name, method_params=None):
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
        perts = fgsm_minimal_perturbations(x, model, sess, params=method_params)
    elif method_name == "jsma":
        jsma = get_crafter(method_name, model, sess, method_params)
        adv_x = jsma.generate_np(x, **method_params)
        perts = adv_x - x
    else:
        pass

    perts_norm = LA.norm(perts, axis=(1, 2))
    assert perts_norm.shape == (len(x), 1), perts_norm.shape
    return np.mean(perts_norm/LA.norm(x))