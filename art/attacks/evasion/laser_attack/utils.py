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

import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Line:
    """
    Representation of the linear function.
    """

    r: float
    b: float

    def __call__(self, x: float) -> float:
        return np.math.tan(self.r) * x + self.b

    def distance_of_point_from_the_line(self, x: float, y: float) -> float:
        """
        Calculate distance between line and point using L2 norm.
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula

        :param x: X coordinate of a point.
        :param y: Y coordinate of a point.
        :returns: Distance.
        """
        y_difference = np.abs(self(x) - y)
        slope_squared = np.math.pow(np.math.tan(self.r), 2)
        return y_difference / np.math.sqrt(1.0 + slope_squared)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.r, self.b])


@dataclass
class Range:
    """
    Representation of mathematical range concept
    """

    left: float
    right: float

    def __contains__(self, value):
        return self.left <= value < self.right

    @property
    def length(self):
        return self.right - self.left


class AdversarialObject(ABC):
    """
    Abstract class that represents an adversarial object placed on an input image in order to attack a neural network.
    """

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: int, y: int) -> Any:
        raise NotImplementedError


class AdvObjectGenerator(ABC):
    """
    Abstract class that define basic behaviours related to generation of an adversarial objects on images.
    """

    min_params: AdversarialObject
    max_params: AdversarialObject

    @abstractmethod
    def update_params(
        self,
        params: AdversarialObject,
        *args,
        **kwargs,
    ) -> AdversarialObject:
        raise NotImplementedError

    @abstractmethod
    def random(self) -> AdversarialObject:
        raise NotImplementedError


class ImageGenerator:
    """
    General class responsible for generation and updating images used to attack neural network. Images are crated basing
    on adversarial objects passed to the class.
    """

    def update_image(self, original_image: np.ndarray, params: AdversarialObject) -> np.ndarray:
        """
        Update original image used for prediction by adding image of the adversarial object to it,
        in order to create adversarial example.

        :param original_image: Image to attack.
        :param params: Adversarial object.
        :returns: Original image with the adversarial object on it.
        """
        image_shape = original_image.shape
        if len(original_image.shape) == 4:
            image_shape = image_shape[1:]

        adv_object_image = self.generate_image(params, image_shape)
        if len(original_image.shape) == 4:
            adv_object_image = np.expand_dims(adv_object_image, 0)
        return self.add_images(original_image, adv_object_image)

    def add_images(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Add two images and return resultant image.

        :param image1: First image.
        :param image2: Second image.
        """
        return add_images(image1, image2)

    def generate_image(self, adv_object: Callable, shape: Tuple) -> np.ndarray:
        """
        Generate image of the adversarial object.

        :param adv_object: Adversarial object.
        :param shape: Shape of the desired image.
        :returns: Image of the adversarial object.
        """
        laser_image = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[0]):
                rgb = adv_object(i, j)
                laser_image[i, j, 0] = np.clip(rgb[0], 0, 1)
                laser_image[i, j, 1] = np.clip(rgb[1], 0, 1)
                laser_image[i, j, 2] = np.clip(rgb[2], 0, 1)

        return laser_image


def wavelength_to_RGB(wavelength: Union[float, int]) -> List[float]:
    """
    Converts wavelength in nanometers to the RGB color.

    :param wavelength: wavelength in the nanometers
    :returns: Array of normalized RGB values.
    """
    wavelength = float(wavelength)
    range1 = Range(380, 440)
    range2 = Range(440, 490)
    range3 = Range(490, 510)
    range4 = Range(510, 580)
    range5 = Range(580, 645)
    range6 = Range(645, 780)

    if wavelength in range1:
        R = (range1.right - wavelength) / range1.length
        G = 0.0
        B = 1.0
        return [R, G, B]
    if wavelength in range2:
        R = 0.0
        G = (wavelength - range2.left) / range2.length
        B = 1.0
        return [R, G, B]
    if wavelength in range3:
        R = 0.0
        G = 1.0
        B = (range3.right - wavelength) / range3.length
        return [R, G, B]
    if wavelength in range4:
        R = (wavelength - range4.left) / range4.length
        G = 1.0
        B = 0.0
        return [R, G, B]
    if wavelength in range5:
        R = 1.0
        G = (range5.right - wavelength) / range5.length
        B = 0.0
        return [R, G, B]
    if wavelength in range6:
        R = 1.0
        G = 0.0
        B = 0.0
        return [R, G, B]

    return [0.0, 0.0, 0.0]


def add_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Add two normalized RGB images and return resultant image.
    If some pixel value exceeds 1, then it's set to 1.

    :param image1: First image.
    :param image2: Second image.
    :returns: Resultant image.
    """
    if image1.shape != image2.shape:
        raise Exception("Wrong size")
    return np.clip(image1 + image2, 0, 1)


def save_NRGB_image(image: np.ndarray, number=0, name_length=5, directory="attack"):
    """
    Saves normalized RGB image, passed as numpy array to the set directory - default: "attack".

    :param image: Image to save.
    :param number: i.e. class of the image.
    :param name_length: Length of the random string in the name.
    :param directory: Directory where images will be saved.
    """
    ALPHABET = np.array(list(string.ascii_letters))
    Path(directory).mkdir(exist_ok=True)
    im_name = f"{directory}/{number}_{''.join(np.random.choice(ALPHABET, size=name_length))}.jpg"
    plt.imsave(im_name, image)


@dataclass
class DebugInfo:
    """
    Logs debug informations during attacking process.
    """

    logger: Logger
    artifacts_directory: str

    def log(self, adv_object: AdversarialObject) -> None:
        """
        Prints debug info on the stderr.

        :param adv_object: Parameters of the adversarial object, printed out to the stderr.
        """
        self.logger.info(adv_object)

    def save_image(self, image: np.ndarray) -> None:
        """
        Saves images generated during lasting process to the artifacts directory.

        :param image: Image to save.
        """
        save_NRGB_image(image, name_length=5, directory=self.artifacts_directory)

    @staticmethod
    def report(instance: 'DebugInfo', adv_object: AdversarialObject, image: np.ndarray) -> None:
        """
        Log info and save image in the preset directory, based on the :instance.

        :param instance: DebugInfo object.
        :param adv_object: Object that will be printed out.
        :param image: Image to save.
        """
        if instance.logger is not None:
            instance.log(adv_object)
        if instance.artifacts_directory is not None:
            instance.save_image(image)


def show_NRGB_image(image: np.ndarray) -> None:
    """
    Plots an image passed as RGB array

    :param image: image to plot
    """
    plt.imshow(image)
