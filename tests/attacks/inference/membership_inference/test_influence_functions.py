from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from art.attacks.inference.membership_inference.self_influence_function_attack import SelfInfluenceFunctionAttack
from art.attacks.inference.membership_inference.influence_functions import (
    calc_s_test,
    calc_self_influence,
)
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier, ScikitlearnClassifier
from art.utils import load_nursery, to_categorical

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

def test_calc_s_test():
    # Create mock PyTorch model and data loaders
    model = create_mock_model()
    test_data = [(torch.rand(10), torch.tensor(0)) for _ in range(50)]
    train_data = [(torch.rand(10), torch.tensor(0)) for _ in range(100)]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)

    # Call the calc_s_test function
    s_tests, save_path = calc_s_test(
        model,
        test_loader,
        train_loader,
        save=False,
        gpu=-1,
        damp=0.01,
        scale=25,
        recursion_depth=5000,
        r=1,
        start=0
    )

    # Assert the results (you may need to modify this based on your specific use case)
    assert len(s_tests) == 50  # Assuming the test_loader has 50 samples
    assert save_path is False  # Since save=False, the save_path should be False

