import pytest
import torch

from over_the_air.paper_equations import firstTemporalDerivative, secondTemporalDerivative, adversarialLoss, thicknessRegularization, roughnessRegularization

# $ pytest -k TestClassDemoInstance


#Dimension Test for Derivatives
class DerivativeTest:
    derivativeInput = [torch.Tensor([1,0,0,0]),
                      torch.Tensor([0,1,0,0]),
                      torch.Tensor([0,0,1,0]),
                      torch.Tensor([0,0,0,1])]

    def test_firstTemporalDerivative(self):
        for x in DerivativeTest.derivativeInput:
            assert firstTemporalDerivative(x).size() == [1,4]
            #if x == DerivativeTest.derivativeInput[0]:
                #assert firstTemporalDerivative(x) == torch.Tensor([0,0,0,0])

    def test_secondTemporalDerivative(self):
        for x in Derivativetest.derivativeInput:
            assert secondTemporalDerivative(x).size() == [1,4]
            #if x == DerivativeTest.derivativeInput:
                #assert secondTemporalDerivative(x) == torch.tensor([0,0,0,0])

#Dimension Test for Regularization
class RegularizationTest:
    regularizationInput = [torch.Tensor([1,0,0,0]),
                            torch.Tensor([0,1,0,0]),
                            torch.Tensor([0,0,1,0]),
                            torch.Tensor([0,0,0,1])]

    def test_thicknessRegularization(self):
        for x in RegularizationTest.regularizationInput:
            assert thicknessRegularization(x, 1).size() == [1,4]

    def test_roughnessRegularizaiton(self):
        for x in RegularizationTest.regularizationInput:
            assert roughnessRegularization(x, 1).size() == [1,4]

