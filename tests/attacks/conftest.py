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
import pytest

from tests.utils import ARTTestFixtureNotImplemented

logger = logging.getLogger(__name__)


@pytest.fixture
def tabular_dl_estimator_for_attack(framework, tabular_dl_estimator):
    def _tabular_dl_estimator_for_attack(attack, clipped=True):
        classifier = tabular_dl_estimator(clipped)
        classifier_list = [classifier]

        classifier_tested = [
            potential_classifier
            for potential_classifier in classifier_list
            if all(t in type(potential_classifier).__mro__ for t in attack._estimator_requirements)
        ]

        if len(classifier_tested) == 0:
            raise ARTTestFixtureNotImplemented(
                "no estimator available", tabular_dl_estimator.__name__, framework, {"attack": attack}
            )
        return classifier_tested[0]

    return _tabular_dl_estimator_for_attack
