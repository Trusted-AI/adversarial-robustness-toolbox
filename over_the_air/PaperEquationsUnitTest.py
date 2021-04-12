import pytest
import torch

from over_the_air.paper_equations import (firstTemporalDerivative, secondTemporalDerivative, adversarialLoss,
    thicknessRegularization, roughnessRegularization)


# $ pytest -k TestClassDemoInstance

# Derivative Tests
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

# Regularization Tests
class RegularizationTest:
    OnesInput = torch.ones(4, 3, 3, 3)
    ZeroesInput = torch.zeros(4, 3, 3, 3)

    def test_thicknessRegularization(self):
        assert thicknessRegularization(self.OnesInput, 1).size() == (4, 3, 3, 3)
        assert thicknessRegularization(self.ZeroesInput, 1).size() == (4, 3, 3, 3)
        # Output Check
        assert thicknessRegularization(self.OnesInput, 1) == self.OnesInput
        assert thicknessRegularization(self.ZeroesInput, 1) == self.ZeroesInput

    def test_roughnessRegularizaiton(self):
        assert roughnessRegularization(self.OnesInput, 1).size() == (4, 3, 3, 3)
        assert roughnessRegularization(self.ZeroesInput, 1).size() == (4, 3, 3, 3)
        # Output Check
        assert roughnessRegularization(self.ZeroesInput, 1) == self.ZeroesInput


"""
class AdversarialLossTest: 
"""


