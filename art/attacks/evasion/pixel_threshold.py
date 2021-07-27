# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the Threshold Attack and Pixel Attack.
The Pixel Attack is a generalisation of One Pixel Attack.

| One Pixel Attack Paper link: https://arxiv.org/ans/1710.08864
| Pixel and Threshold Attack Paper link: https://arxiv.org/abs/1906.06026
"""
# pylint: disable=C0302
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from itertools import product
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# Currently, a modified version of SciPy's differential evolution is used in
# code. An ideal version would be using the import as follows,
# from scipy.optimize import differential_evolution
# In the meantime, the modified implementation is used which is defined in the
# lines `453-1457`.
# Otherwise may use Tensorflow's implementation of DE.

from six import string_types
from scipy._lib._util import check_random_state
from scipy.optimize.optimize import _status_message
from scipy.optimize import OptimizeResult, minimize
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class PixelThreshold(EvasionAttack):
    """
    These attacks were originally implemented by Vargas et al. (2019) & Su et al.(2019).

    | One Pixel Attack Paper link: https://arxiv.org/abs/1710.08864
    | Pixel and Threshold Attack Paper link: https://arxiv.org/abs/1906.06026
    """

    attack_params = EvasionAttack.attack_params + ["th", "es", "max_iter", "targeted", "verbose", "verbose_es"]
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        th: Optional[int] = None,
        es: int = 0,
        max_iter: int = 100,
        targeted: bool = False,
        verbose: bool = True,
        verbose_es: bool = False,
    ) -> None:
        """
        Create a :class:`.PixelThreshold` instance.

        :param classifier: A trained classifier.
        :param th: threshold value of the Pixel/ Threshold attack. th=None indicates finding a minimum threshold.
        :param es: Indicates whether the attack uses CMAES (0) or DE (1) as Evolutionary Strategy.
        :param max_iter: Sets the Maximum iterations to run the Evolutionary Strategies for optimisation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param verbose: Print verbose messages of ES and show progress bars.
        """
        super().__init__(estimator=classifier)

        self._project = True
        self.type_attack = -1
        self.th = th  # pylint: disable=C0103
        self.es = es  # pylint: disable=C0103
        self.max_iter = max_iter
        self._targeted = targeted
        self.verbose = verbose
        self.verbose_es = verbose_es
        PixelThreshold._check_params(self)

        if self.estimator.channels_first:
            self.img_rows = self.estimator.input_shape[-2]
            self.img_cols = self.estimator.input_shape[-1]
            self.img_channels = self.estimator.input_shape[-3]
        else:
            self.img_rows = self.estimator.input_shape[-3]
            self.img_cols = self.estimator.input_shape[-2]
            self.img_channels = self.estimator.input_shape[-1]

    def _check_params(self) -> None:
        if self.th is not None:
            if self.th <= 0:
                raise ValueError("The perturbation size `eps` has to be positive.")

        if not isinstance(self.es, int):
            raise ValueError("The flag `es` has to be of type int.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The flag `verbose` has to be of type bool.")

        if not isinstance(self.verbose_es, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
        if self.estimator.clip_values is None:
            raise ValueError("This attack requires estimator clip values to be defined.")

    def rescale_input(self, x):
        """Rescale inputs"""
        x = x.astype(ART_NUMPY_DTYPE) / 255.0
        x = (x * (self.estimator.clip_values[1] - self.estimator.clip_values[0])) + self.estimator.clip_values[0]
        return x

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param max_iter: Maximum number of optimisation iterations.
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes, return_one_hot=False)

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = np.argmax(self.estimator.predict(x), axis=1)
        else:
            if self.estimator.nb_classes == 2 and y.shape[1] == 1:
                raise ValueError(
                    "This attack has not yet been tested for binary classification with a single output classifier."
                )
            if len(y.shape) > 1:
                y = np.argmax(y, axis=1)

        if self.th is None:
            logger.info(
                "Performing minimal perturbation Attack. \
                This could take long time to process. \
                For sanity check, pass th=10 to the Attack instance."
            )

        # NOTE: Pixel and Threshold Attacks are well defined for unprocessed images where the pixel values are,
        #       8-Bit color i.e., the pixel values are np.uint8 in range [0, 255].

        # TO-DO: Better checking of input image.
        #        All other cases not tested needs the images to be rescaled to [0, 255].
        if self.estimator.clip_values[1] != 255.0:
            self.rescale = True
            x = (x - self.estimator.clip_values[0]) / (self.estimator.clip_values[1] - self.estimator.clip_values[0])
            x = x * 255.0

        x = x.astype(ART_NUMPY_DTYPE)

        adv_x_best = []
        self.adv_th = []
        for image, target_class in tqdm(zip(x, y), desc="Pixel threshold", disable=not self.verbose):

            if self.th is None:

                min_th = -1
                start, end = 1, 127

                image_result = image

                while True:

                    threshold = (start + end) // 2
                    success, trial_image_result = self._attack(image, target_class, threshold)

                    if success:
                        image_result = trial_image_result
                        end = threshold - 1
                        min_th = threshold
                    else:
                        start = threshold + 1

                    if end < start:
                        break

                self.adv_th = [min_th]

            else:

                success, image_result = self._attack(image, target_class, self.th)

                if not success:
                    image_result = image

            adv_x_best += [image_result]

        adv_x_best_array = np.array(adv_x_best)

        if y is not None:
            y = to_categorical(y, self.estimator.nb_classes)

        if self.rescale:
            x = self.rescale_input(x)
            adv_x_best_array = self.rescale_input(adv_x_best_array)

        logger.info(
            "Success rate of Attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, adv_x_best_array, self.targeted, 1),
        )
        return adv_x_best_array

    def _get_bounds(self, img: np.ndarray, limit) -> Tuple[List[list], list]:
        """
        Define the bounds for the image `img` within the limits `limit`.
        """

        def bound_limit(value):
            return np.clip(value - limit, 0, 255), np.clip(value + limit, 0, 255)

        minbounds, maxbounds, bounds, initial = [], [], [], []

        for i, j, k in product(range(img.shape[-3]), range(img.shape[-2]), range(img.shape[-1])):
            temp = img[i, j, k]
            initial += [temp]
            bound = bound_limit(temp)
            if self.es == 0:
                minbounds += [bound[0]]
                maxbounds += [bound[1]]
            else:
                bounds += [bound]
        if self.es == 0:
            bounds = [minbounds, maxbounds]

        return bounds, initial

    def _perturb_image(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:  # pylint: disable=W0613,R0201
        """
        Perturbs the given image `img` with the given perturbation `x`.
        """
        return img

    def _attack_success(self, adv_x, x, target_class):
        """
        Checks whether the given perturbation `adv_x` for the image `img` is successful.
        """
        adv = self._perturb_image(adv_x, x)

        if self.rescale:
            adv = self.rescale_input(adv)

        predicted_class = np.argmax(self.estimator.predict(adv)[0])
        return bool(
            (self.targeted and predicted_class == target_class)
            or (not self.targeted and predicted_class != target_class)
        )

    def _attack(self, image: np.ndarray, target_class: np.ndarray, limit: int) -> Tuple[bool, np.ndarray]:
        """
        Attack the given image `image` with the threshold `limit` for the `target_class` which is true label for
        untargeted attack and targeted label for targeted attack.
        """
        bounds, initial = self._get_bounds(image, limit)

        def predict_fn(x):
            adv = self._perturb_image(x, image)

            if self.rescale:
                adv = self.rescale_input(adv)

            predictions = self.estimator.predict(adv)[:, target_class]
            return predictions if not self.targeted else 1 - predictions

        def callback_fn(x, convergence=None):  # pylint: disable=R1710,W0613
            if self.es == 0:
                if self._attack_success(x.result[0], image, target_class):
                    raise CMAEarlyStoppingException("Attack Completed :) Earlier than expected")
            else:
                return self._attack_success(x, image, target_class)

        if self.es == 0:
            from cma import CMAOptions

            opts = CMAOptions()
            if not self.verbose_es:
                opts.set("verbose", -9)
                opts.set("verb_disp", 40000)
                opts.set("verb_log", 40000)
                opts.set("verb_time", False)

            opts.set("bounds", bounds)

            if self.type_attack == 0:
                std = 63
            else:
                std = limit

            from cma import CMAEvolutionStrategy

            strategy = CMAEvolutionStrategy(initial, std / 4, opts)

            try:
                strategy.optimize(
                    predict_fn,
                    maxfun=max(1, 400 // len(bounds)) * len(bounds) * 100,
                    callback=callback_fn,
                    iterations=self.max_iter,
                )
            except CMAEarlyStoppingException as err:
                if self.verbose_es:
                    logger.info(err)

            adv_x = strategy.result[0]
        else:
            strategy = differential_evolution(
                predict_fn,
                bounds,
                disp=self.verbose_es,
                maxiter=self.max_iter,
                popsize=max(1, 400 // len(bounds)),
                recombination=1,
                atol=-1,
                callback=callback_fn,
                polish=False,
            )
            adv_x = strategy.x

        if self._attack_success(adv_x, image, target_class):
            return True, self._perturb_image(adv_x, image)[0]

        return False, image


class PixelAttack(PixelThreshold):
    """
    This attack was originally implemented by Vargas et al. (2019). It is generalisation of One Pixel Attack originally
    implemented by Su et al. (2019).

    | One Pixel Attack Paper link: https://arxiv.org/abs/1710.08864
    | Pixel Attack Paper link: https://arxiv.org/abs/1906.06026
    """

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        th: Optional[int] = None,
        es: int = 1,
        max_iter: int = 100,
        targeted: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Create a :class:`.PixelAttack` instance.

        :param classifier: A trained classifier.
        :param th: threshold value of the Pixel/ Threshold attack. th=None indicates finding a minimum threshold.
        :param es: Indicates whether the attack uses CMAES (0) or DE (1) as Evolutionary Strategy.
        :param max_iter: Sets the Maximum iterations to run the Evolutionary Strategies for optimisation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param verbose: Indicates whether to print verbose messages of ES used.
        """
        super().__init__(classifier, th, es, max_iter, targeted, verbose)
        self.type_attack = 0

    def _perturb_image(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Perturbs the given image `img` with the given perturbation `x`.
        """
        if x.ndim < 2:
            x = np.array([x])
        imgs = np.tile(img, [len(x)] + [1] * (x.ndim + 1))
        x = x.astype(int)
        for adv, image in zip(x, imgs):
            for pixel in np.split(adv, len(adv) // (2 + self.img_channels)):
                x_pos, y_pos, *rgb = pixel
                if not self.estimator.channels_first:
                    image[x_pos % self.img_rows, y_pos % self.img_cols] = rgb
                else:
                    image[:, x_pos % self.img_rows, y_pos % self.img_cols] = rgb
        return imgs

    def _get_bounds(self, img: np.ndarray, limit) -> Tuple[List[list], list]:
        """
        Define the bounds for the image `img` within the limits `limit`.
        """
        initial: List[np.ndarray] = []
        bounds: List[List[int]]
        if self.es == 0:
            for count, (i, j) in enumerate(product(range(self.img_rows), range(self.img_cols))):
                initial += [i, j]
                for k in range(self.img_channels):
                    if not self.estimator.channels_first:
                        initial += [img[i, j, k]]
                    else:
                        initial += [img[k, i, j]]

                if count == limit - 1:
                    break

            min_bounds = [0, 0]
            for _ in range(self.img_channels):
                min_bounds += [0]
            min_bounds = min_bounds * limit
            max_bounds = [self.img_rows, self.img_cols]
            for _ in range(self.img_channels):
                max_bounds += [255]
            max_bounds = max_bounds * limit
            bounds = [min_bounds, max_bounds]
        else:
            bounds = [[0, self.img_rows], [0, self.img_cols]]
            for _ in range(self.img_channels):
                bounds += [[0, 255]]
            bounds = bounds * limit
        return bounds, initial


class ThresholdAttack(PixelThreshold):
    """
    This attack was originally implemented by Vargas et al. (2019).

    | Paper link: https://arxiv.org/abs/1906.06026
    """

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        th: Optional[int] = None,
        es: int = 0,
        max_iter: int = 100,
        targeted: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Create a :class:`.PixelThreshold` instance.

        :param classifier: A trained classifier.
        :param th: threshold value of the Pixel/ Threshold attack. th=None indicates finding a minimum threshold.
        :param es: Indicates whether the attack uses CMAES (0) or DE (1) as Evolutionary Strategy.
        :param max_iter: Sets the Maximum iterations to run the Evolutionary Strategies for optimisation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param verbose: Indicates whether to print verbose messages of ES used.
        """
        super().__init__(classifier, th, es, max_iter, targeted, verbose)
        self.type_attack = 1

    def _perturb_image(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Perturbs the given image `img` with the given perturbation `x`.
        """
        if x.ndim < 2:
            x = x[None, ...]
        imgs = np.tile(img, [len(x)] + [1] * (x.ndim + 1))
        x = x.astype(int)
        for adv, image in zip(x, imgs):
            for count, (i, j, k) in enumerate(
                product(
                    range(image.shape[-3]),
                    range(image.shape[-2]),
                    range(image.shape[-1]),
                )
            ):
                image[i, j, k] = adv[count]
        return imgs


class CMAEarlyStoppingException(Exception):
    """Raised when CMA is stopping early after successful optimisation."""

    pass


# TODO: Make the attack compatible with current version of SciPy Optimize
# Differential Evolution
# pylint: disable=W0105
"""
A slight modification to Scipy's implementation of differential evolution.
To speed up predictions, the entire parameters array is passed to `self.func`,
where a neural network model can batch its computations and execute in parallel
Search for `CHANGES` to find all code changes.

Dan Kondratyuk 2018

Original code adapted from
https://github.com/scipy/scipy/blob/70e61dee181de23fdd8d893eaa9491100e2218d7/scipy/optimize/_differentialevolution.py
----------
differential_evolution:The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""

# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
# Copyright (c) 2003-2017 SciPy Developers.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of Enthought nor the names of the SciPy Developers
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# SciPy bundles a number of libraries that are compatibly licensed.  We list
# these here.
# Name: Numpydoc
# Files: doc/sphinxext/numpydoc/*
# License: 2-clause BSD
#   For details, see doc/sphinxext/LICENSE.txt
# Name: scipy-sphinx-theme
# Files: doc/scipy-sphinx-theme/*
# License: 3-clause BSD, PSF and Apache 2.0
#   For details, see doc/sphinxext/LICENSE.txt
# Name: Six
# Files: scipy/_lib/six.py
# License: MIT
#   For details, see the header inside scipy/_lib/six.py
# Name: Decorator
# Files: scipy/_lib/decorator.py
# License: 2-clause BSD
#   For details, see the header inside scipy/_lib/decorator.py
# Name: ID
# Files: scipy/linalg/src/id_dist/*
# License: 3-clause BSD
#   For details, see scipy/linalg/src/id_dist/doc/doc.tex
# Name: L-BFGS-B
# Files: scipy/optimize/lbfgsb/*
# License: BSD license
#   For details, see scipy/optimize/lbfgsb/README
# Name: SuperLU
# Files: scipy/sparse/linalg/dsolve/SuperLU/*
# License: 3-clause BSD
#   For details, see scipy/sparse/linalg/dsolve/SuperLU/License.txt
# Name: ARPACK
# Files: scipy/sparse/linalg/eigen/arpack/ARPACK/*
# License: 3-clause BSD
#   For details, see scipy/sparse/linalg/eigen/arpack/ARPACK/COPYING
# Name: Qhull
# Files: scipy/spatial/qhull/*
# License: Qhull license (BSD-like)
#   For details, see scipy/spatial/qhull/COPYING.txt
# Name: Cephes
# Files: scipy/special/cephes/*
# License: 3-clause BSD
#   Distributed under 3-clause BSD license with permission from the author,
#   see https://lists.debian.org/debian-legal/2004/12/msg00295.html
#   Cephes Math Library Release 2.8:  June, 2000
#   Copyright 1984, 1995, 2000 by Stephen L. Moshier
#   This software is derived from the Cephes Math Library and is
#   incorporated herein by permission of the author.
#   All rights reserved.
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimer in the
#         documentation and/or other materials provided with the distribution.
#       * Neither the name of the <organization> nor the
#         names of its contributors may be used to endorse or promote products
#         derived from this software without specific prior written permission.
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Name: Faddeeva
# Files: scipy/special/Faddeeva.*
# License: MIT
#   Copyright (c) 2012 Massachusetts Institute of Technology
#   Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the
#   "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to
#   the following conditions:
#   The above copyright notice and this permission notice shall be
#   included in all copies or substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
#   LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#   OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__all__ = ["differential_evolution"]

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(
    func,
    bounds,
    args=(),
    strategy="best1bin",
    maxiter=1000,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=False,
    polish=True,
    init="latinhypercube",
    atol=0,
):
    """Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient
    methods) to find the minimium, and can search large areas of candidate
    space, but often requires larger numbers of function evaluations than
    conventional gradient based techniques.
    The algorithm is due to Storn and Price [1]_.
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.RandomState` singleton is used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with seed.
        If `seed` is already a `np.random.RandomState instance`, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population subset could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.  If `polish`
        was employed, and a lower minimum was obtained by the polishing, then
        OptimizeResult also contains the ``jac`` attribute.
    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the
    population the algorithm mutates each candidate solution by mixing with
    other candidate solutions to create a trial candidate. There are several
    strategies [2]_ for creating trial candidates, which suit some problems
    more than others. The 'best1bin' strategy is a good starting point for many
    systems. In this strategy two members of the population are randomly
    chosen. Their difference is used to mutate the best member (the `best` in
    `best1bin`), :math:`b_0`,
    so far:
    .. math::
        b' = b_0 + mutation * (population[rand0] - population[rand1])
    A trial vector is then constructed. Starting with a randomly chosen 'i'th
    parameter the trial is sequentially filled (in modulo) with parameters from
    `b'` or the original candidate. The choice of whether to use `b'` or the
    original candidate is made with a binomial distribution (the 'bin' in
    'best1bin') - a random number in [0, 1) is generated.  If this number is
    less than the `recombination` constant then the parameter is loaded from
    `b'`, otherwise it is loaded from the original candidate.  The final
    parameter is always loaded from `b'`.  Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.
    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.
    .. versionadded:: 0.15.0
    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.
    >>> from scipy.optimize import rosen, differential_evolution
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = differential_evolution(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)
    Next find the minimum of the Ackley function
    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).
    >>> from scipy.optimize import differential_evolution
    >>> import numpy as np
    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi *x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = differential_evolution(ackley, bounds)
    >>> result.x, result.fun
    (array([ 0.,  0.]), 4.4408920985006262e-16)
    References
    ----------
    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
    .. [3] http://en.wikipedia.org/wiki/Differential_evolution
    """

    solver = DifferentialEvolutionSolver(
        func,
        bounds,
        args=args,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=polish,
        callback=callback,
        disp=disp,
        init=init,
        atol=atol,
    )
    return solver.solve()


class DifferentialEvolutionSolver:
    """This class implements the differential evolution solver
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.random.RandomState` singleton is
        used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with `seed`.
        If `seed` is already a `np.random.RandomState` instance, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
        is used to polish the best population member at the end. This requires
        a few more function evaluations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population could be used, for
        example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    """

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {
        "best1bin": "_best1",
        "randtobest1bin": "_randtobest1",
        "currenttobest1bin": "_currenttobest1",
        "best2bin": "_best2",
        "rand2bin": "_rand2",
        "rand1bin": "_rand1",
    }
    _exponential = {
        "best1exp": "_best1",
        "rand1exp": "_rand1",
        "randtobest1exp": "_randtobest1",
        "currenttobest1exp": "_currenttobest1",
        "best2exp": "_best2",
        "rand2exp": "_rand2",
    }

    __init_error_msg = (
        "The population initialization method must be one of "
        "'latinhypercube' or 'random', or an array of shape "
        "(M, N) where N is the number of parameters and M>5"
    )

    def __init__(
        self,
        func,
        bounds,
        args=(),
        strategy="best1bin",
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        maxfun=np.inf,
        callback=None,
        disp=False,
        polish=True,
        init="latinhypercube",
        atol=0,
    ):

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        self.polish = polish

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if not np.all(np.isfinite(mutation)) or np.any(np.array(mutation) >= 2) or np.any(np.array(mutation) < 0):
            raise ValueError(
                "The mutation constant must be a float in "
                "U[0, 2), or specified as a tuple(min, max)"
                " where min < max and min, max are in U[0, 2)."
            )

        self.dither = None
        if hasattr(mutation, "__iter__") and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype="float").T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError(
                "bounds should be a sequence containing " "real valued (min, max) pairs for each value" " in x"
            )

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        self.num_population_members = max(5, popsize * self.parameter_count)

        self.population_shape = (self.num_population_members, self.parameter_count)

        self._nfev = 0
        if isinstance(init, string_types):
            if init == "latinhypercube":
                self.init_population_lhs()
            elif init == "random":
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)

        self.disp = disp

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the
        # following size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (
            segsize * rng.random_sample(self.population_shape)
            # Offset each segment to cover the entire parameter range
            # [0, 1)
            + np.linspace(0.0, 1.0, self.num_population_members, endpoint=False)[:, np.newaxis]
        )

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = np.ones(self.num_population_members) * np.inf

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)

        # reset population energies
        self.population_energies = np.ones(self.num_population_members) * np.inf

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initialises the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (M, len(x)), where len(x) is the number of parameters.
            The population is clipped to the lower and upper `bounds`.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if np.size(popn, 0) < 5 or popn.shape[1] != self.parameter_count or len(popn.shape) != 2:
            raise ValueError("The population supplied needs to have shape" " (M, len(x)), where M > 4.")

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members, self.parameter_count)

        # reset population energies
        self.population_energies = np.ones(self.num_population_members) * np.inf

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return np.std(self.population_energies) / np.abs(np.mean(self.population_energies) + _MACHEPS)

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes. If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message["success"]

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        # do the optimisation.
        for nit in range(1, self.maxiter + 1):
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message["maxfev"]
                break

            if self.disp:
                print("differential_evolution step %d: f(x)= %g" % (nit, self.population_energies[0]))

            # should the solver terminate?
            convergence = self.convergence

            if (
                self.callback
                and self.callback(
                    self._scale_parameters(self.population[0]),
                    convergence=self.tol / convergence,
                )
                is True
            ):
                warning_flag = True
                status_message = "callback function requested stop early by returning True"
                break

            intol = np.std(self.population_energies) <= self.atol + self.tol * np.abs(np.mean(self.population_energies))
            if warning_flag or intol:
                break

        else:
            status_message = _status_message["maxiter"]
            warning_flag = True

        de_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True),
        )

        if self.polish:
            result = minimize(
                self.func,
                np.copy(de_result.x),
                method="L-BFGS-B",
                bounds=self.limits.T,
                args=self.args,
            )

            self._nfev += result.nfev
            de_result.nfev = self._nfev

            if result.fun < de_result.fun:
                de_result.fun = result.fun
                de_result.x = result.x
                de_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        return de_result

    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """

        ##############
        # CHANGES: self.func operates on the entire parameters array
        ##############
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates])  # TODO: can be vectorized
        energies = self.func(parameters, *self.args)
        self.population_energies = energies
        self._nfev += itersize

        # for index, candidate in enumerate(self.population):
        #     if self._nfev > self.maxfun:
        #         break

        #     parameters = self._scale_parameters(candidate)
        #     self.population_energies[index] = self.func(parameters,
        #                                                 *self.args)
        #     self._nfev += 1

        ##############
        ##############

        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self

    def __next__(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        if self.dither is not None:
            self.scale = self.random_number_generator.rand() * (self.dither[1] - self.dither[0]) + self.dither[0]

        ##############
        # CHANGES: self.func operates on the entire parameters array
        ##############

        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))
        trials = np.array([self._mutate(c) for c in range(itersize)])  # TODO:can be vectorized
        for trial in trials:
            self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in trials])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize

        for candidate, (energy, trial) in enumerate(zip(energies, trials)):
            # if the energy of the trial candidate is lower than the
            # original population member then replace it
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy

                # if the trial candidate also has a lower energy than the
                # best solution then replace that as well
                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial

        # for candidate in range(self.num_population_members):
        #     if self._nfev > self.maxfun:
        #         raise StopIteration

        #     # create a trial solution
        #     trial = self._mutate(candidate)

        #     # ensuring that it's in the range [0, 1)
        #     self._ensure_constraint(trial)

        #     # scale from [0, 1) to the actual parameter value
        #     parameters = self._scale_parameters(trial)

        #     # determine the energy of the objective function
        #     energy = self.func(parameters, *self.args)
        #     self._nfev += 1

        #     # if the energy of the trial candidate is lower than the
        #     # original population member then replace it
        #     if energy < self.population_energies[candidate]:
        #         self.population[candidate] = trial
        #         self.population_energies[candidate] = energy

        #         # if the trial candidate also has a lower energy than the
        #         # best solution then replace that as well
        #         if energy < self.population_energies[0]:
        #             self.population_energies[0] = energy
        #             self.population[0] = trial

        ##############
        ##############

        return self.x, self.population_energies[0]

    def next(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        for index in np.where((trial < 0) | (trial > 1))[0]:
            trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate):  # pylint: disable=R1710
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.randint(0, self.parameter_count)

        if self.strategy in ["currenttobest1exp", "currenttobest1bin"]:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        if self.strategy in self._exponential:
            i = 0
            while i < self.parameter_count and rng.rand() < self.cross_over_probability:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r_0, r_1 = samples[:2]
        return self.population[0] + self.scale * (self.population[r_0] - self.population[r_1])

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r_0, r_1, r_2 = samples[:3]
        return self.population[r_0] + self.scale * (self.population[r_1] - self.population[r_2])

    def _randtobest1(self, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r_0, r_1, r_2 = samples[:3]
        bprime = np.copy(self.population[r_0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r_1] - self.population[r_2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """
        currenttobest1bin, currenttobest1exp
        """
        r_0, r_1 = samples[:2]
        bprime = self.population[candidate] + self.scale * (
            self.population[0] - self.population[candidate] + self.population[r_0] - self.population[r_1]
        )
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r_0, r_1, r_2, r_3 = samples[:4]
        bprime = self.population[0] + self.scale * (
            self.population[r_0] + self.population[r_1] - self.population[r_2] - self.population[r_3]
        )

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r_0, r_1, r_2, r_3, r_4 = samples
        bprime = self.population[r_0] + self.scale * (
            self.population[r_1] + self.population[r_2] - self.population[r_3] - self.population[r_4]
        )

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs
