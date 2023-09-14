# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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


@pytest.mark.skip_framework("tensorflow1", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf")
def test_attack_repr(image_dl_estimator):

    from art.attacks.evasion import ProjectedGradientDescentNumpy

    classifier, _ = image_dl_estimator(from_logits=True)

    attack = ProjectedGradientDescentNumpy(
        estimator=classifier,
        targeted=True,
        decay=0.5,
    )
    print(repr(attack))
    assert repr(attack) == (
        "ProjectedGradientDescentNumpy(norm=inf, eps=0.3, eps_step=0.1, targeted=True, "
        + "num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=0.5, "
        + "max_iter=100, random_eps=False, verbose=True, )"
    )
