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

import logging
from typing import List, Optional, Tuple, Union, Callable

import numpy as np
from numpy.lib.arraysetops import isin
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.laser_attack.algorithms import greedy_search
from art.attacks.evasion.laser_attack.utils import (AdversarialObject,
                                                    AdvObjectGenerator,
                                                    DebugInfo, ImageGenerator,
                                                    Line, wavelength_to_RGB)

logger = logging.getLogger(__name__)

# %%
class LaserAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "optimizer",
        "iterations",
        "random_initializations",
        "tensor_board, bool] = False",
        "actual_class",
        "actual_class_confidence"
    ]
    _estimator_requirements = ()

    def __init__(
        self,
        estimator,
        iterations: int,
        laser_generator: AdvObjectGenerator,
        image_generator: ImageGenerator = ImageGenerator(),
        random_initializations: int = 1,
        optimisation_algorithm: Callable = greedy_search,
        tensor_board: Union[str, bool] = False,
        debug: Optional[DebugInfo] = None
    ) -> None:
        """
        :param estimator: Predictor of the image class.
        :param iterations: Maximum number of iterations of the algorithm.
        :param laser_generator: Object responsible for
            generation laser beams images and their updation.
        :param image_generator: Object responsible for image generation.
        :param random_initializations: How many times repeat the attack.
        :param tensor_board:
        :param debug: Optional debug handler.
        """

        super().__init__(estimator=estimator, tensor_board=tensor_board)
        self.iterations = iterations
        self.random_initializations = random_initializations
        self.optimisation_algorithm = optimisation_algorithm
        self._laser_generator = laser_generator
        self._image_generator = image_generator
        self._debug = debug

        self._check_params()

    def generate(
        self,
        x: np.ndarray,
        *args,
        **kwargs
    ) -> Optional[List]:
        """

        Generate adversarial example for a single image.

        :param x: Image to attack as a tensor (NRGB = (1, ...))
        :return: List of paris of adversarial objects and predicted class.
        """

        adversarial_params = []
        for image_index in range(x.shape[0]):
            params, adv_class = self._generate_for_single_input(x[image_index])
            adversarial_params.append((params, adv_class))

        return adversarial_params


    def _generate_for_single_input(
        self,
        x: np.ndarray,
        *args,
        **kwargs
    ) -> Tuple[Optional[AdversarialObject], Optional[int]]:
        """
        Generate adversarial example for a single image.

        :param x: Image to attack as a tensor (NRGB = (1, ...))
        """

        image = np.expand_dims(x, 0)
        prediction = self.estimator.predict(image)
        actual_class = prediction.argmax()
        actual_class_confidence = prediction[0][actual_class]

        for _ in range(self.random_initializations):
            laser_params, predicted_class = self.generate_parameters(
                image,
                (actual_class, actual_class_confidence)
            )
            if laser_params is not None:
                logger.info("Found adversarial params: %s", laser_params)
                return laser_params, predicted_class
        logger.warning("Couldn't find adversarial laser parameters")

        return None, None

    def _check_params(self) -> None:
        super()._check_params()
        if self.iterations <= 0:
            raise ValueError("The iterations number has to be positive.")
        if self.random_initializations <= 0:
            raise ValueError("The random initializations has to be positive.")

    def generate_parameters(
        self,
        image: np.ndarray,
        actual_prediction: Tuple[int, float]
    ) -> Tuple[Optional[AdversarialObject], Optional[int]]:
        """
        Generate adversarial parameters and wrong class predicted by the
        neural network.

        :param image: Image to attack.
        :param actual_prediction: Correct class of the image and prediction.
        :returns: Pair of adversarial parameters and predicted class.
        """

        actual_class, confidence = actual_prediction
        return self.optimisation_algorithm(
            image=image,
            estimator=self.estimator,
            iterations=self.iterations,
            actual_class=actual_class,
            actual_class_confidence=confidence,
            adv_object_generator=self._laser_generator,
            image_generator=self._image_generator,
            debug=self._debug,
        )

class LaserBeam(AdversarialObject):

    def __init__(self, wavelength: float, width: float, line: Line):
        """
        :param wavelength:
        :param width:
        :param line:
        """
        self.wavelength = float(wavelength)
        self.line = line
        self.width = float(width)
        self.rgb = np.array(wavelength_to_RGB(self.wavelength))

    def __call__(self, x: int, y: int) -> np.ndarray:
        """
        Generate pixel of a laser beam.

        :param x: X coordinate of a pixel.
        :param y: Y coordinate of a pixel.
        :returns: List of 3 normalized RGB values that represents a pixel.
        """
        _x, _y = float(x), float(y)
        distance = self.line.distance_of_point_from_the_line(_x, _y)

        if distance <= self.width / 2.:
            return self.rgb
        if self.width/2. <= distance <= 5*self.width:
            return (
                np.math.sqrt(self.width) /
                np.math.pow(distance, 2) *
                self.rgb
            )

        return np.array([0., 0., 0.])

    def __repr__(self) -> str:
        return f"LaserBeam(wavelength={self.wavelength}, Line={str(self.line)}, width={self.width})"

    @staticmethod
    def from_numpy(theta: np.ndarray) -> 'LaserBeam':
        """
        :param theta: List of the laser beam parameters, passed as List
            int the order: wavelength[nm], slope, bias[pixels], width[pixels]
        :returns: New class object based on :theta.
        """
        return LaserBeam(
            wavelength=theta[0],
            line=Line(theta[1], theta[2]),
            width=theta[3],
        )

    @staticmethod
    def from_array(theta: List) -> 'LaserBeam':
        """
        Create instance of the class using parameters :theta.

        :param theta: List of the laser beam parameters, passed as List
            int the order: wavelength[nm], slope, bias[pixels], width[pixels].
        :returns: New class object based on :theta.
        """
        return LaserBeam(
            wavelength=theta[0],
            line=Line(theta[1], theta[2]),
            width=theta[3],
        )

    def to_numpy(self):
        line = self.line
        return np.array([
            self.wavelength,
            line.r,
            line.b,
            self.width
        ])

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return LaserBeam.from_numpy(other * self.to_numpy())
        if isinstance(other, np.ndarray):
            return LaserBeam.from_numpy(self.to_numpy() * other)
        if isinstance(other, list):
            return LaserBeam.from_numpy(self.to_numpy() * np.array(other))

    def __rmul__(self, other):
        return self * other

class LaserBeamGenerator(AdvObjectGenerator):
    """
    Generate LaserBeam objects for the
    LaserBeamAttack purpose

    """
    def __init__(
        self,
        min_params: LaserBeam,
        max_params: LaserBeam,
        max_step: float=20/100
    ) -> None:
        """
        :params min_params: left bound of the params range
        :params max_params: right bound of the params range
        :params max_step: maximal part of the random LaserBeam
            object drawed from the range.
        """
        self.min_params = min_params
        self.max_params = max_params
        self.max_step = max_step
        self.__params_ranges = max_params.to_numpy() - min_params.to_numpy()

    def update_params(
        self,
        params: LaserBeam,
        *args,
        **kwargs,
    ) -> LaserBeam:
        """
        Updates parameters of the received LaserBeam object
        in the random direction.

        :param params: LaserBeam object to be updated.
        """
        sign = kwargs.get("sign", 1)
        random_step = np.random.uniform(0, self.max_step)
        d_params = self.__params_ranges * random_step * self._random_direction()
        theta_prim = LaserBeam.from_numpy(
            params.to_numpy() + sign * d_params
        )
        theta_prim = self.clip(theta_prim)
        return theta_prim

    def _random_direction(self) -> np.ndarray:
        """
        Generate random array of ones that will decide which parameters
            of a laser beam will be updated: wavelength, angle, bias, width.

        :returns: random array of ones
        """
        Q = np.asfarray([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [1,1,0,0],
            [1,0,1,0],
            [1,0,0,1],
            [0,1,1,0],
            [0,1,0,1],
            [0,0,1,1]
        ])

        mask = Q[np.random.choice(len(Q))]
        return mask

    def clip(self, params: LaserBeam):
        """
        Keep received parameters in the tolerance ranges.

        :param params: Parameters of the LaserBeam
            that will be eventually clipped.
        :return: LaserBeam parameters in the desired ranges.
        """
        clipped_params = np.clip(
            params.to_numpy(),
            self.min_params.to_numpy(),
            self.max_params.to_numpy()
        )
        params.wavelength = clipped_params[0]
        params.line.r = clipped_params[1]
        params.line.b = clipped_params[2]
        params.width = clipped_params[3]
        return params

    def random(self) -> LaserBeam:
        """
        Generate object of the LaserBeam class
        that will have randomly generated parameters
        in the tolerance ranges.

        :return: LaserBeam object with random parameters
        """
        random_params = (
            self.min_params.to_numpy()
            + np.random.uniform(0, 1)
            * (self.max_params.to_numpy() - self.min_params.to_numpy())
        )

        return LaserBeam.from_numpy(random_params)

class LaserBeamAttack(LaserAttack):
    """
    Implementation of the `LaserBeam` attack.

    | Paper link: https://openaccess.thecvf.com/content/CVPR2021/papers/Duan_Adversarial_Laser_Beam_Effective_Physical-World_Attack_to_DNNs_in_a_CVPR_2021_paper.pdf
    """

    def __init__(
        self,
        estimator,
        iterations: int,
        max_laser_beam: \
            Union[LaserBeam, Tuple[float, float, float, int]],
        min_laser_beam: \
            Union[LaserBeam, Tuple[float, float, float, int]] = (380., 0., 1., 1),
        random_initializations: int = 1,
        image_generator: ImageGenerator = ImageGenerator(),
        tensor_board: Union[str, bool] = False,
        debug: Optional[DebugInfo] = None
    ) -> None:
        """
        :param estimator: Predictor of the image class.
        :param iterations: Maximum number of iterations of the algorithm.
        :param max_laser_beam: LaserBeam with maximal parameters or
            tuple (wavelength, angle::radians, bias, width) of the laser parameters.
        :param min_laser_beam: LaserBeam with minimal parameters or
            tuple (wavelength, angle::radians, bias, width) of the laser parameters.
        :param image_generator: Object responsible for image generation.
        :param random_initializations: How many times repeat the attack.
        :param tensor_board:
        :param debug: Optional debug handler.
        """

        if isinstance(min_laser_beam, Tuple):
            min_laser_beam= LaserBeam.from_array(list(min_laser_beam))
        if isinstance(max_laser_beam, Tuple):
            max_laser_beam= LaserBeam.from_array(list(max_laser_beam))

        super().__init__(
            estimator,
            iterations,
            LaserBeamGenerator(min_laser_beam, max_laser_beam),
            image_generator=image_generator,
            random_initializations=random_initializations,
            tensor_board=tensor_board,
            debug=debug
        )
