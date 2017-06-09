import config

import numpy as np
import numpy.linalg as LA

from src.attackers.fast_gradient import FastGradientMethod

supported_methods = {"fgsm": {"class": FastGradientMethod,
                              "params": {"eps_step": 0.1, "eps_max": 1., "clip_min": 0., "clip_max": 1.}},
                      # "jsma": {"class": SaliencyMapMethod,
                      #          "params": {"theta": 1., "gamma": 0.01, "clip_min": 0., "clip_max": 1.}}
                      }


def get_crafter(method, model, session, params=None):

    try:
        crafter = supported_methods[method]["class"](model, sess=session)
    except:
        raise NotImplementedError("{} crafting method not supported.".format(method))

    if params:
        crafter.set_params(**params)
    else:
        crafter.set_params(**supported_methods[method]["params"])

    return crafter


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

    crafter = get_crafter(method_name, model, sess, method_params)
    adv_x = crafter.generate(x, minimal=True, **method_params)

    perts_norm = LA.norm((adv_x-x).reshape(x.shape[0], -1), ord=crafter.ord, axis=1)

    assert perts_norm.shape == (len(x), ), perts_norm.shape

    return np.mean(perts_norm/LA.norm(x))
