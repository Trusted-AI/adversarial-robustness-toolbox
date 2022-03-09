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
This module implements the `LaserAttack` attack.

| Paper link: https://arxiv.org/abs/2103.06504
"""

import logging
from typing import Callable, List, Optional, Tuple, Union, Any

import numpy as np

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.laser_attack.algorithms import greedy_search
from art.attacks.evasion.laser_attack.utils import (
    AdversarialObject,
    AdvObjectGenerator,
    DebugInfo,
    ImageGenerator,
    Line,
    wavelength_to_rgb,
)

logger = logging.getLogger(__name__)


class LaserAttack(EvasionAttack):
    """
    Implementation of a generic laser attack case.
    """

    attack_params = EvasionAttack.attack_params + [
        "iterations",
        "laser_generator",
        "image_generator",
        "random_initializations",
        "optimisation_algorithm",
        "debug",
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
        debug: Optional[DebugInfo] = None,
    ) -> None:
        """
        :param estimator: Predictor of the image class.
        :param iterations: Maximum number of iterations of the algorithm.
        :param laser_generator: Object responsible for generation laser beams images and their update.
        :param image_generator: Object responsible for image generation.
        :param random_initializations: How many times repeat the attack.
        :param optimisation_algorithm: Algorithm used to generate adversarial example. May be replaced.
        :param debug: Optional debug handler.
        """

        super().__init__(estimator=estimator)
        self.iterations = iterations
        self.random_initializations = random_initializations
        self.optimisation_algorithm = optimisation_algorithm
        self._laser_generator = laser_generator
        self._image_generator = image_generator
        self._debug = debug

        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples.

        :param x: Images to attack as a tensor in NHWC order
        :param y: Array of correct classes
        :return: Array of adversarial images
        """

        if x.ndim != 4:  # pragma: no cover
            raise ValueError("Unrecognized input dimension. Only tensors NHWC are acceptable.")

        parameters = self.generate_parameters(x, y)
        adversarial_images = np.zeros_like(x)
        for image_index in range(x.shape[0]):
            laser_params, _ = parameters[image_index]
            if laser_params is None:
                adversarial_images[image_index] = x[image_index]
                continue
            adversarial_image = self._image_generator.update_image(x[image_index], laser_params)
            adversarial_images[image_index] = adversarial_image

        return adversarial_images

    def generate_parameters(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> List[Tuple[Optional[AdversarialObject], Optional[int]]]:
        """
        Generate adversarial parameters for given images.

        :param x: Images to attack as a tensor (NRGB = (1, ...))
        :param y: Correct classes
        :return: List of tuples of adversarial objects and predicted class.
        """
        result = []
        for image_index in range(x.shape[0]):
            laser_params, adv_class = self._generate_params_for_single_input(
                x[image_index], y[image_index] if y is not None else None
            )
            result.append((laser_params, adv_class))
        return result

    def _generate_params_for_single_input(
        self, x: np.ndarray, y: Optional[int] = None
    ) -> Tuple[Optional[AdversarialObject], Optional[int]]:
        """
        Generate adversarial example params for a single image.

        :param x: Image to attack as a tensor (NRGB = (1, ...))
        :param y: Correct class of the image. If not provided, it is set to the prediction of the model.
        :return: Adversarial object params and adversarial class number.
        """

        image = np.expand_dims(x, 0)
        prediction = self.estimator.predict(image)
        if y is not None:
            actual_class = y
        else:
            actual_class = prediction.argmax()
        actual_class_confidence = prediction[0][actual_class]

        for _ in range(self.random_initializations):
            laser_params, predicted_class = self._attack_single_image(image, actual_class, actual_class_confidence)
            if laser_params is not None:
                logger.info("Found adversarial params: %s", laser_params)
                return laser_params, predicted_class
        logger.warning("Couldn't find adversarial laser parameters")

        return None, None

    def _check_params(self) -> None:
        super()._check_params()
        if self.estimator.channels_first:
            raise ValueError("Channels first models are not supported. Supported tensor format: NHWC")
        if self.iterations <= 0:
            raise ValueError("The iterations number has to be positive.")
        if self.random_initializations <= 0:
            raise ValueError("The random initializations has to be positive.")

    def _attack_single_image(
        self, x: np.ndarray, y: int, confidence: float
    ) -> Tuple[Optional[AdversarialObject], Optional[int]]:
        """
        Attack particular image with given class.

        :param x: Image to attack.
        :param y: Correct class of the image.
        :returns: Pair of adversarial parameters and predicted class.
        """

        return self.optimisation_algorithm(
            image=x,
            estimator=self.estimator,
            iterations=self.iterations,
            actual_class=y,
            actual_class_confidence=confidence,
            adv_object_generator=self._laser_generator,
            image_generator=self._image_generator,
            debug=self._debug,
        )


class LaserBeam(AdversarialObject):
    """
    Representation of the attacking object used in the paper.
    """

    def __init__(self, wavelength: float, width: float, line: Line):
        """
        :param wavelength: Wavelength in nanometers of the laser beam.
        :param width: Width of the laser beam in pixels.
        :param line: Line object used to determine shape of the laser beam.
        """
        self.wavelength = float(wavelength)
        self.line = line
        self.width = float(width)
        self.rgb = np.array(wavelength_to_rgb(self.wavelength))

    def __call__(self, x: int, y: int) -> np.ndarray:
        """
        Generate pixel of a laser beam.

        :param x: X coordinate of a pixel.
        :param y: Y coordinate of a pixel.
        :returns: List of 3 normalized RGB values (between 0 and 1) that represents a pixel.
        """
        _x, _y = float(x), float(y)
        distance = self.line.distance_of_point_from_the_line(_x, _y)

        if distance <= self.width / 2.0:
            return self.rgb
        if self.width / 2.0 <= distance <= 5 * self.width:
            return np.math.sqrt(self.width) / np.math.pow(distance, 2) * self.rgb  # type: ignore

        return np.array([0.0, 0.0, 0.0])

    def __repr__(self) -> str:
        return f"LaserBeam(wavelength={self.wavelength}, Line={str(self.line)}, width={self.width})"

    @staticmethod
    def from_numpy(theta: np.ndarray) -> "LaserBeam":
        """
        :param theta: List of the laser beam parameters, passed as List int the order:
            wavelength[nm], slope angle[radians], bias[pixels], width[pixels].
        :returns: New class object based on :theta.
        """
        return LaserBeam(
            wavelength=theta[0],
            line=Line(theta[1], theta[2]),
            width=theta[3],
        )

    @staticmethod
    def from_array(theta: List) -> "LaserBeam":
        """
        Create instance of the class using parameters :theta.

        :param theta: List of the laser beam parameters, passed as List int the order:
            wavelength[nm], slope angle[radians], bias[pixels], width[pixels].
        :returns: New class object based on :theta.
        """
        return LaserBeam(
            wavelength=theta[0],
            line=Line(theta[1], theta[2]),
            width=theta[3],
        )

    def to_numpy(self) -> np.ndarray:
        line = self.line
        return np.array([self.wavelength, line.angle, line.bias, self.width])

    def __mul__(self, other: Union[float, int, list, np.ndarray]) -> "LaserBeam":
        if isinstance(other, (float, int)):
            return LaserBeam.from_numpy(other * self.to_numpy())
        if isinstance(other, np.ndarray):
            return LaserBeam.from_numpy(self.to_numpy() * other)
        if isinstance(other, list):
            return LaserBeam.from_numpy(self.to_numpy() * np.array(other))
        raise Exception("Not accepted value.")

    def __rmul__(self, other) -> "LaserBeam":
        return self * other


class LaserBeamGenerator(AdvObjectGenerator):
    """
    Generator of the LaserBeam objects for the LaserBeamAttack purpose
    """

    def __init__(self, min_params: LaserBeam, max_params: LaserBeam, max_step: float = 20 / 100) -> None:
        """
        :params min_params: left bound of the params range
        :params max_params: right bound of the params range
        :params max_step: maximal part of the random LaserBeam object drawn from the range.
        """
        self.min_params = min_params
        self.max_params = max_params
        self.max_step = max_step
        self.__params_ranges = max_params.to_numpy() - min_params.to_numpy()

    def update_params(self, params: Any, **kwargs) -> LaserBeam:
        """
        Updates parameters of the received LaserBeam object in the random direction.

        :param params: LaserBeam object to be updated.
        :returns: Updated object.
        """
        sign = kwargs.get("sign", 1)
        random_step = np.random.uniform(0, self.max_step)
        d_params = self.__params_ranges * random_step * self._random_direction()
        theta_prim = LaserBeam.from_numpy(params.to_numpy() + sign * d_params)
        theta_prim = self.clip(theta_prim)
        return theta_prim

    @staticmethod
    def _random_direction() -> np.ndarray:
        """
        Generate random array of ones that will decide which parameters of a laser beam will be updated:
            wavelength, angle, bias and width.

        :returns: Random array of ones (mask).
        """
        q_mask = np.asfarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
            ]
        )

        mask = q_mask[np.random.choice(len(q_mask))]
        return mask

    def clip(self, params: LaserBeam):
        """
        Keep received parameters in the tolerance ranges.

        :param params: Parameters of the LaserBeam that will be eventually clipped.
        :return: LaserBeam parameters in the desired ranges.
        """
        clipped_params = np.clip(params.to_numpy(), self.min_params.to_numpy(), self.max_params.to_numpy())
        params.wavelength = clipped_params[0]
        params.line.angle = clipped_params[1]
        params.line.bias = clipped_params[2]
        params.width = clipped_params[3]
        return params

    def random(self) -> LaserBeam:
        """
        Generate object of the LaserBeam class that will have randomly generated parameters in the tolerance ranges.

        :return: LaserBeam object with random parameters
        """
        random_params = self.min_params.to_numpy() + np.random.uniform(0, 1) * (
            self.max_params.to_numpy() - self.min_params.to_numpy()
        )

        return LaserBeam.from_numpy(random_params)


class LaserBeamAttack(LaserAttack):
    """
    Implementation of the `LaserBeam` attack.

    | Paper link: https://arxiv.org/abs/2103.06504
    """

    def __init__(
        self,
        estimator,
        iterations: int,
        max_laser_beam: Union[LaserBeam, Tuple[float, float, float, int]],
        min_laser_beam: Union[LaserBeam, Tuple[float, float, float, int]] = (380.0, 0.0, 1.0, 1),
        random_initializations: int = 1,
        image_generator: ImageGenerator = ImageGenerator(),
        debug: Optional[DebugInfo] = None,
    ) -> None:
        """
        :param estimator: Predictor of the image class.
        :param iterations: Maximum number of iterations of the algorithm.
        :param max_laser_beam: LaserBeam with maximal parameters or tuple (wavelength, angle::radians, bias, width)
            of the laser parameters.
        :param min_laser_beam: LaserBeam with minimal parameters or tuple (wavelength, angle::radians, bias, width)
            of the laser parameters.
        :param image_generator: Object responsible for image generation.
        :param random_initializations: How many times repeat the attack.
        :param debug: Optional debug handler.
        """

        if isinstance(min_laser_beam, tuple):
            min_laser_beam_obj = LaserBeam.from_array(list(min_laser_beam))
        else:
            min_laser_beam_obj = min_laser_beam
        if isinstance(max_laser_beam, tuple):
            max_laser_beam_obj = LaserBeam.from_array(list(max_laser_beam))
        else:
            max_laser_beam_obj = max_laser_beam

        super().__init__(
            estimator,
            iterations,
            LaserBeamGenerator(min_laser_beam_obj, max_laser_beam_obj),
            image_generator=image_generator,
            random_initializations=random_initializations,
            debug=debug,
        )
