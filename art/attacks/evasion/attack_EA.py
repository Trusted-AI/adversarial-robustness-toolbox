import math
import numpy as np
from sys import stdout
import random
random.seed(0)

"""
This module implements the Evolutionary Algorithm based Adversarial Attack (attack_EA). 

| Paper link: https://www.sciencedirect.com/science/article/pii/S1568494623004155
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

class EA(EvasionAttack):
    """
    This class implements the black-box attack `attack_EA`.

     Paper link: https://www.sciencedirect.com/science/article/pii/S1568494623004155
    """
    attack_params = EvasionAttack.attack_params + [
        "max_iter",
        "confidence",
        "targeted",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin) # ?????
    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        max_iter: int = 10000,
        confidence: float = 0.51,
        targeted: bool = False,
    ):
        """
        Create an attack_EA  attack instance.

        :param classifier: A trained classifier predicting probabilities and not logits.
        :param max_iter: The maximum number of iterations.
        :param confidence: Confidence of adversarial examples
        :param targeted: perform targeted attack
        """
        super().__init__(estimator=classifier)

        self.max_iter = max_iter
        self.confidence = confidence
        self.targeted = targeted
        self.pop_size = 40
        self.numberOfElites = 10

    def _get_class_prob(self, preds: np.ndarray, class_no: np.array) -> np.ndarray:
        '''
        :param preds: an array of predictions of individuals for all the categories: (40, 1000) shaped array
        :param class_no: for the targeted attack target category index number; for the untargeted attack ancestor category index number
        :return: an array of the prediction of individuals only for the target/ancestor category: (40,) shaped  array
        '''
        return preds[:, class_no]

    def _get_fitness(self, probs: np.ndarray) -> np.ndarray:
        '''
         It simply returns the CNN's probability for the images but different objective functions can be used here.
        :param probs: an array of images' probabilities of selected CNN
        :return: returns images' probabilities in an array (40,)
        '''
        fitness = probs
        return fitness

    def _selection_untargeted(self, images: np.ndarray, fitness: np.ndarray):
        '''
        Population will be divided into elite, middle_class, and didn't make it based on
        images (individuals) fitness values. The images furthest from the ancestor category will be closer to be in the elite.
        :param images: the population of images in an array: size (pop_size, 224, 224, 3)
        :param fitness: an array of images' propabilities of selected CNN
        :return: returns a tuple of elite, middle_class images, fitness values of elites, index number of elites
                in the population array, and random_keep images as numpy arrays.
        '''
        idx_elite = fitness.argsort()[:self.numberOfElites]
        elite_fitness = fitness[idx_elite]
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[self.numberOfElites: int(half_pop_size)]
        elite = images[idx_elite, :]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0] / 2 - self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, elite_fitness, idx_elite, random_keep

    def _selection_targeted(self, images: np.ndarray, fitness: np.ndarray):
        '''
        Population will be divided into elite, middle_class, and didn't make it based on
        images (individuals) fitness values. The images closest to the target category will be closer to be in the elite.
        :param images: the population of images in an array: size (pop_size, 224, 224, 3)
        :param fitness: an array of images' probabilities of selected CNN
        :return: returns elite, middle_class images, fitness values of elites, index number of elites
                in the population array, and random_keep images as numpy arrays.
        '''
        idx_elite = fitness.argsort()[-self.numberOfElites:]
        elite_fitness = fitness[idx_elite]
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[int(half_pop_size):-self.numberOfElites]
        elite = images[idx_elite, :][::-1]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0] / 2 - self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, elite_fitness, idx_elite, random_keep

    def _get_no_of_pixels(self, im_size: int) -> int:
        '''
        :param im_size: Original inputs' size, represented by an integer value.
        :return: returns an integer that will be used to decide how many pixels will be mutated in the image during the current generation.
        '''
        u_factor = np.random.uniform(0.0, 1.0)
        n = 60  # normally 60, the smaller n -> more pixels to mutate
        res = (u_factor ** (1.0 / (n + 1))) * im_size
        no_of_pixels = im_size - res
        return no_of_pixels

    def _mutation(self, x: np.ndarray, no_of_pixels: int, mutation_group: np.ndarray, percentage: float,
                  boundary_min: int, boundary_max: int) -> np.ndarray:
        '''
        :param x: An array with the original input to be attacked.
        :param no_of_pixels: An integer determines the number of pixels to mutate in the original input for the current
            generation.
        :param mutation_group: An array with the individuals which will be mutated
        :param percentage: A decimal number from 0 to 1 that represents the percentage of individuals in the mutation
            group that will undergo mutation.
        :param boundary_min: keep the pixel within [0, 255]
        :param boundary_max: keep the pixel within [0, 255]
        :return: An array of mutated individuals
        '''
        mutated_group = mutation_group.copy()
        random.shuffle(mutated_group)
        no_of_individuals = len(mutated_group)
        for individual in range(int(no_of_individuals * percentage)):
            locations_x = np.random.randint(x.shape[0], size=int(no_of_pixels))
            locations_y = np.random.randint(x.shape[1], size=int(no_of_pixels))
            locations_z = np.random.randint(x.shape[2], size=int(no_of_pixels))
            new_values = random.choices(np.array([-1, 1]), k=int(no_of_pixels))
            mutated_group[individual, locations_x, locations_y, locations_z] = mutated_group[
                                                                                   individual, locations_x, locations_y, locations_z] - new_values
        mutated_group = np.clip(mutated_group, boundary_min, boundary_max)
        return mutated_group

    def _get_crossover_parents(self, crossover_group: np.ndarray) -> list:
        size = crossover_group.shape[0]
        no_of_parents = random.randrange(0, size, 2)
        parents_idx = random.sample(range(0, size), no_of_parents)
        return parents_idx

    def _crossover(self, x: np.ndarray, crossover_group: np.ndarray, parents_idx: list) -> np.ndarray:
        crossedover_group = crossover_group.copy()
        for i in range(0, len(parents_idx), 2):
            parent_index_1 = parents_idx[i]
            parent_index_2 = parents_idx[i + 1]
            crossover_range = int(x.shape[0] * 0.15)  # 15% of the image will be crossovered.
            size_x = np.random.randint(0, crossover_range)
            start_x = np.random.randint(0, x.shape[0] - size_x)
            size_y = np.random.randint(0, crossover_range)
            start_y = np.random.randint(0, x.shape[1] - size_y)
            z = np.random.randint(x.shape[2])
            temp = crossedover_group[parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y, z]
            crossedover_group[parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y,
            z] = crossedover_group[
                 parent_index_2,
                 start_x: start_x + size_x,
                 start_y: start_y + size_y,
                 z]
            crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z] = temp
        return crossedover_group

    def _generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        '''
        :param x: An array with the original inputs to be attacked.
        :param y: An integer with the true or target labels.
        :return: An array holding the adversarial examples.
        '''
        boundary_min = 0
        boundary_max = 255

        img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2])).copy()
        img = self.estimator.preprocess_input(img)   # ??????????????????
        preds = self.estimator.predict(img)
        label0 = self.estimator.decode_predictions(preds)    # ??????????????????
        label1 = label0[0][0]  # Gets the Top1 label and values for reporting.
        ancestor = label1[1]  # label
        ancIndx = np.argmax(preds)
        print(f'Before the image is:  {ancestor} --> {label1[2]} ____ index: {ancIndx}')
        if self.targeted:
            print('Target class index number is: ', y)
        images = np.array([x] * self.pop_size).astype(int)  # pop_size * ancestor images are created
        count = 0
        while True:
            img = self.estimator.preprocess_input(images)    # ??????????????????
            preds = self.estimator.predict(img)  # predictions of 40 images
            domIndx = np.argmax(preds[int(np.argmax(preds) / 1000)])
            # Dominant category report ##################
            label0 = self.estimator.decode_predictions(preds)  # Reports predictions with label and label values   # ??????????????
            label1 = label0[0][0]  # Gets the Top1 label and values for reporting.
            domCat = label1[1]   # label
            domCat_prop = label1[2]  # label probability
            ###########################################

            percentage_middle_class = 1
            percentage_keep = 1


        # Select population classes based on fitness
            if self.targeted:
                probs = self._get_class_prob(preds, y)
                fitness = self._get_fitness(probs)
                elite, middle_class, elite_fitness, idx_elite, random_keep = self._selection_targeted(images, fitness)
            else:
                probs = self._get_class_prob(preds, ancIndx)
                fitness = self._get_fitness(probs)
                elite, middle_class, elite_fitness, idx_elite, random_keep = self._selection_untargeted(images, fitness)
            elite2 = elite.copy()
            keep = np.concatenate((elite2, random_keep))

        # Reproduce individuals by mutating Elite and Middle class---------
            # mutate and crossover individuals
            im_size = x.shape[0] * x.shape[1] * x.shape[2]
            no_of_pixels = self._get_no_of_pixels(im_size)
            mutated_middle_class = self._mutation(x, no_of_pixels, middle_class, percentage_middle_class, boundary_min,
                                                  boundary_max)
            mutated_keep_group1 = self._mutation(x, no_of_pixels, keep, percentage_keep, boundary_min, boundary_max)
            mutated_keep_group2 = self._mutation(x, no_of_pixels, mutated_keep_group1, percentage_keep, boundary_min,
                                                 boundary_max)

            all_ = np.concatenate((mutated_middle_class, mutated_keep_group2))
            parents_idx = self._get_crossover_parents(all_)
            crossover_group = self._crossover(x, all_, parents_idx)

        # Create new population
            images = np.concatenate((elite, crossover_group))
        # Report the progress on the screen
            stdout.write(f'\rgeneration: {count}/{self.max_iter} ______ {domCat}: {domCat_prop} ____ index: {domIndx}')
            count += 1

        # Terminate the algorithm if:
            if count == self.max_iter:
                # if algorithm can not create the adversarial image within "max_iter" stop the algorithm.
                print(f"Failed to generate adversarial image within {self.max_iter} generations")
                break
            if not self.targeted and domIndx != ancIndx:
                # if the attack is not targeted, the algorithm stops as soon as the image is classified in a category other than its true category.
                break
            if self.targeted and domIndx == y and domCat_prop > self.confidence:
               # if the attack is targeted, the algorithm stops when the image is classified in the target category y.
                break
        return images[0]










