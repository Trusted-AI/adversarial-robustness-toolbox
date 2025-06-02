# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2024
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
import numpy as np

from art.evaluations.evaluation import Evaluation


class GreatScorePyTorch(Evaluation):

    def __init__(self, classifier):
        """
        Calculate the GREAT score and accuracy for given samples using the specified classifier.

        :param classifier: ART Classifier of the model to evaluate.
        """
        self.classifier = classifier
        super().__init__()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[list[float], float]:
        """
        Calculate the GREAT score and accuracy for given samples using the specified model.

        :param x: Input samples (images) as a numpy array.
        :param y: True labels for the samples.

        :return: great_score, accuracy
        """
        # Calculate accuracy
        y_pred: np.ndarray = self.classifier.predict(x=x)
        correct_predictions: np.ndarray = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
        accuracy = np.mean(correct_predictions)

        # Calculate the GREAT score
        k = 2
        top2_indices = np.argpartition(y_pred, -k, axis=1)[:, -k:]  # (batch_size, 2), unsorted
        top2_values = np.take_along_axis(y_pred, top2_indices, axis=1)  # corresponding values
        # Now sort these top-2 values and indices in descending order
        sorted_order = np.argsort(top2_values, axis=1)[:, ::-1]
        top2_values_sorted = np.take_along_axis(top2_values, sorted_order, axis=1)
        great_scores = np.where(correct_predictions, top2_values_sorted[:, 0] - top2_values_sorted[:, 1], 0)
        average_great_score = np.mean(great_scores)

        return average_great_score, accuracy
