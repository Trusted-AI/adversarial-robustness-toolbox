# %%
from abc import abstractmethod, ABC
import string
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Any
from dataclasses import dataclass
from logging import Logger
from pathlib import Path


# %%
@dataclass
class Line:
    r: float
    b: float

    def __call__(self, x: float) -> float:
        return np.math.tan(self.r) * x + self.b

    # L2 norm.
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula
    def distance_of_point_from_the_line(self, x: float, y: float) -> float:
        y_difference = np.abs(self(x) - y)
        slope_squared = np.math.pow(np.math.tan(self.r), 2)
        return y_difference / np.math.sqrt(1. + slope_squared)

    def to_numpy(self):
        return np.array([self.r, self.b])


# %%
@dataclass
class Range:
    left: float
    right: float

    def __contains__(self, value):
        return self.left <= value < self.right

    @property
    def length(self):
        return self.right - self.left


# %%
class AdversarialObject(ABC):

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: int, y: int) -> Any:
        raise NotImplementedError


class AdvObjectGenerator(ABC):
    min_params: AdversarialObject
    max_params: AdversarialObject

    @abstractmethod
    def update_params(
        self,
        params: AdversarialObject,
        *args,
        **kwargs,
    ) -> AdversarialObject:
        """

        """
        raise NotImplementedError

    @abstractmethod
    def random(self) -> AdversarialObject:
        raise NotImplementedError


class ImageGenerator:

    def update_image(
        self,
        original_image: np.ndarray,
        params: AdversarialObject
    ) -> np.ndarray:
        laser_image = self.generate_image(params, original_image.shape[1:])
        return self.add_images(original_image, np.expand_dims(laser_image, 0))

    def add_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> np.ndarray:
        return add_images(image1, image2)

    def generate_image(
        self,
        adv_object: AdversarialObject,
        shape
    ) -> np.ndarray:
        laser_image = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[0]):
                rgb =  adv_object(i,j)
                laser_image[i,j,0] = np.clip(rgb[0], 0, 1)
                laser_image[i,j,1] = np.clip(rgb[1], 0, 1)
                laser_image[i,j,2] = np.clip(rgb[2], 0, 1)

        return laser_image
# %%
def wavelength_to_RGB(wavelength: Union[float, int]) -> List[float]:
    wavelength = float(wavelength)
    range1 = Range(380, 440)
    range2 = Range(440, 490)
    range3 = Range(490, 510)
    range4 = Range(510, 580)
    range5 = Range(580, 645)
    range6 = Range(645, 780)

    if wavelength in range1:
        R = (range1.right - wavelength) / range1.length
        G = 0.
        B = 1.
        return [R, G, B]
    if wavelength in range2:
        R = 0.
        G = (wavelength - range2.left) / range2.length
        B = 1.
        return [R, G, B]
    if wavelength in range3:
        R = 0.
        G = 1.
        B = (range3.right - wavelength) / range3.length
        return [R, G, B]
    if wavelength in range4:
        R = (wavelength - range4.left) / range4.length
        G = 1.
        B = 0.
        return [R, G, B]
    if wavelength in range5:
        R = 1.
        G = (range5.right - wavelength) / range5.length
        B = 0.
        return [R, G, B]
    if wavelength in range6:
        R = 1.
        G = 0.
        B = 0.
        return [R, G, B]

    return [0., 0., 0.]

# %%
def add_images(image1, image2):
    if image1.shape != image2.shape:
        raise Exception("Wrong size")
    return np.clip(image1 + image2, 0, 1)

# %%
def add_laser_to_image2(image, laser):
    image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i,j].astype("float")/255. + laser(j,i)
            pixel = np.clip(pixel, 0.0, 1.0)
            pixel = pixel * 255
            image[i,j] = pixel.astype("uint8")
    return image

# %%
def save_NRGB_image(image: np.ndarray, number=0, name_length=5, directory="attack"):
    ALPHABET = np.array(list(string.ascii_letters))
    Path(directory).mkdir(exist_ok=True)
    im_name =f"{directory}/{number}_{''.join(np.random.choice(ALPHABET, size=name_length))}.jpg"
    plt.imsave(im_name, image)

@dataclass
class DebugInfo:
    logger: Logger
    artifacts_directory: str

    def log(self, adv_object: AdversarialObject):
        self.logger.info(adv_object)

    def save_image(self, image: np.ndarray):
        save_NRGB_image(image, name_length=5, directory=self.artifacts_directory)

    @staticmethod
    def report(instance, adv_object: AdversarialObject, image: np.ndarray):
        if instance.logger is not None:
            instance.log(adv_object)
        if instance.artifacts_directory is not None:
            instance.save_image(image)

def show_NRGB_image(image: np.ndarray):
    plt.imshow(image)