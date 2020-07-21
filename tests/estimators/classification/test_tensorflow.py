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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("tensorflow")
def test_set_learning(is_tf_version_2, get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    if not is_tf_version_2:
        assert classifier._feed_dict == {}
        classifier.set_learning_phase(False)
        assert classifier._feed_dict[classifier._learning] is False
        classifier.set_learning_phase(True)
        assert classifier._feed_dict[classifier._learning]
        assert classifier.learning_phase


if __name__ == "__main__":
    pytest.cmdline.main("-q {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
