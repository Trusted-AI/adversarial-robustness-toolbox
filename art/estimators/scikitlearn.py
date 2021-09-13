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
This module implements the abstract estimator for scikit-learn models.
"""
import logging
from typing import Optional, Tuple

from art.estimators.estimator import BaseEstimator

logger = logging.getLogger(__name__)


class ScikitlearnEstimator(BaseEstimator):
    """
    Estimator class for scikit-learn models.
    """

    def _get_input_shape(self, model) -> Optional[Tuple[int, ...]]:
        _input_shape: Optional[Tuple[int, ...]]
        if hasattr(model, "n_features_"):
            _input_shape = (model.n_features_,)
        elif hasattr(model, "n_features_in_"):
            _input_shape = (model.n_features_in_,)
        elif hasattr(model, "feature_importances_"):
            _input_shape = (len(model.feature_importances_),)
        elif hasattr(model, "coef_"):
            if len(model.coef_.shape) == 1:
                _input_shape = (model.coef_.shape[0],)
            else:
                _input_shape = (model.coef_.shape[1],)
        elif hasattr(model, "support_vectors_"):
            _input_shape = (model.support_vectors_.shape[1],)
        elif hasattr(model, "steps"):
            _input_shape = self._get_input_shape(model.steps[0][1])
        else:
            logger.warning("Input shape not recognised. The model might not have been fitted.")
            _input_shape = None
        return _input_shape
