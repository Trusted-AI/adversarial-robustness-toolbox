# %%
from typing import Optional, Tuple, Union, List
import logging
import numpy as np
from art.attacks.evasion.laser_attack.utils import \
    AdversarialObject, AdvObjectGenerator, ImageGenerator, \
    DebugInfo, Line, wavelength_to_RGB
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.laser_attack.algorithms import greedy_search

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
    AdvParams = Tuple[Optional[AdversarialObject], Optional[int]]
    _estimator_requirements = ()

    def __init__(
        self,
        estimator,
        iterations: int,
        laser_generator: AdvObjectGenerator,
        image_generator: ImageGenerator = ImageGenerator(),
        random_initializations: int = 1,
        optimisation_algorithm = greedy_search,
        tensor_board: Union[str, bool] = False,
        debug: Optional[DebugInfo] = None
    ) -> None:

        super().__init__(estimator=estimator, tensor_board=tensor_board)
        self.iterations = iterations
        self.random_initializations = random_initializations
        self.optimisation_algorithm = optimisation_algorithm
        self.__laser_generator = laser_generator
        self.__image_generator = image_generator
        self.__debug = debug

        self._check_params()

    def generate(
        self,
        x: np.ndarray,
        *args,
        **kwargs
    ) -> Optional[List]:

        adversarial_params = []
        for image_index in range(x.shape[0]):
            params, adv_class = self.__generate_for_single_input(x[image_index])
            adversarial_params.append((params, adv_class))

        return adversarial_params


    def __generate_for_single_input(
        self,
        x: np.ndarray,
        *args,
        **kwargs
    ) -> Tuple[Optional[AdversarialObject], Optional[int]]:

        image = np.expand_dims(x, 0)
        prediction = self.estimator.predict(image)
        actual_class = prediction.argmax()
        actual_class_confidence = prediction[0][actual_class]

        for _ in range(self.random_initializations):
            laser_params, predicted_class = self.attack(
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

    def attack(
        self,
        image: np.ndarray,
        actual_prediction: Tuple[int, float]
    ) -> Tuple[Optional[AdversarialObject], Optional[int]]:

        actual_class, confidence = actual_prediction
        # return self.optimisation_algorithm(
        return greedy_search(
            image=image,
            estimator=self.estimator,
            iterations=self.iterations,
            actual_class=actual_class,
            actual_class_confidence=confidence,
            adv_object_generator=self.__laser_generator,
            image_generator=self.__image_generator,
            debug=self.__debug,
        )

# %%
class LaserBeam(AdversarialObject):

    def __init__(self, wavelength: float, width: float, line: Line):
        self.wavelength = float(wavelength)
        self.line = line
        self.width = float(width)
        self.rgb = np.array(wavelength_to_RGB(self.wavelength))

    def __call__(self, x: int, y: int) -> np.ndarray:
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
    def from_numpy(theta: np.ndarray):
        return LaserBeam(
            wavelength=theta[0],
            line=Line(theta[1], theta[2]),
            width=theta[3],
        )

    @staticmethod
    def from_array(theta: List):
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

# %%
class LaserBeamGenerator(AdvObjectGenerator):

    def __init__(
        self,
        min_params: LaserBeam,
        max_params: LaserBeam,
        max_step: float=20/100
    ) -> None:
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
        sign = kwargs.get("sign", 1)
        random_step = np.random.uniform(0, self.max_step)
        d_params = self.__params_ranges * random_step * self._random_direction()
        theta_prim = LaserBeam.from_numpy(
            params.to_numpy() + sign * d_params
        )
        theta_prim = self.clip(theta_prim)
        return theta_prim

    def _random_direction(self) -> np.ndarray:
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
        random_params = (
            self.min_params.to_numpy()
            + np.random.uniform(0, 1)
            * (self.max_params.to_numpy() - self.min_params.to_numpy())
        )

        return LaserBeam.from_numpy(random_params)

# %%
class LaserBeamAttack(LaserAttack):

    def __init__(
        self,
        estimator,
        iterations: int,
        laser_generator: LaserBeamGenerator,
        image_generator: ImageGenerator = ImageGenerator(),
        random_initializations: int = 1,
        tensor_board: Union[str, bool] = False,
        debug: Optional[DebugInfo] = None
    ) -> None:
        super().__init__(
            estimator,
            iterations,
            laser_generator,
            image_generator=image_generator,
            random_initializations=random_initializations,
            tensor_board=tensor_board,
            debug=debug
        )