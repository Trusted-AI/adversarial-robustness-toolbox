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
import pytest

from art.evaluations.great_score import GreatScorePyTorch

from tests.utils import ARTTestException, get_cifar10_image_classifier_pt


@pytest.mark.only_with_platform("pytorch")
def test_great_score(art_warning, image_dl_estimator, get_mnist_dataset):
    try:
        classifier, _ = image_dl_estimator(from_logits=False)
        (_, _), (x_test, y_test) = get_mnist_dataset

        great_score = GreatScorePyTorch(classifier=classifier)
        score, accuracy = great_score.evaluate(x=x_test, y=y_test)
        assert score == pytest.approx(0.03501499, rel=1e-6)
        assert accuracy == pytest.approx(0.2487, rel=1e-4)

    except ARTTestException as e:
        art_warning(e)
