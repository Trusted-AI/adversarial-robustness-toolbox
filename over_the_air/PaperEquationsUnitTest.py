import pytest
import torch

from over_the_air.paper_equations import (firstTemporalDerivative, secondTemporalDerivative, adversarialLoss,
    thicknessRegularization, roughnessRegularization)


# $ pytest -k TestClassDemoInstance

# Dimension Test for Derivatives
class DerivativeTest:
    OnesInput = torch.ones(4, 3, 3, 3)
    ZeroesInput = torch.zeros(4, 3, 3, 3)

    def test_firstTemporalDerivative(self):
        # Dimension Check
        assert firstTemporalDerivative(self.OnesInput).size() == (4, 3, 3, 3)
        assert firstTemporalDerivative(self.ZeroesInput).size() == (4, 3, 3, 3)
        # Output Check
        # Should Output 4 * 3 * 3 * 3 tensor of zeroes
        assert firstTemporalDerivative(self.OnesInput) == self.ZeroesInput

    def test_secondTemporalDerivative(self):
        assert secondTemporalDerivative(self.OnesInput).size() == (4, 3, 3, 3)
        assert secondTemporalDerivative(self.ZeroesInput).size() == (4, 3, 3, 3)
        # Output Check
        # Should Output 4 * 3 * 3 * 3 tensor of zeroes
        assert secondTemporalDerivative(self.ZeroesInput) == self.ZeroesInput

"""
# Dimension Test for Regularization
class RegularizationTest:

    def test_thicknessRegularization(self):
        for x in RegularizationTest.regularizationInput:
            assert thicknessRegularization(x, 1).size() == [1, 4]

    def test_roughnessRegularizaiton(self):
        for x in RegularizationTest.regularizationInput:
            assert roughnessRegularization(x, 1).size() == [1, 4]
"""
