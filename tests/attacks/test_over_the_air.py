import pytest
import torch
# Need random seed for Objective Func
import random
# Need numpy arrays to check values inside torch tensors
import numpy

# import all functions that need to be tested
from over_the_air.paper_equations import (firstTemporalDerivative, secondTemporalDerivative, objectiveFunc, adversarialLoss,
    thicknessRegularization, roughnessRegularization)


# $ pytest -k Test Derivative

# Derivative Tests
class TestDerivative:
    # tensor of ones and zeros
    OnesInput = torch.ones(4, 3, 3, 3)
    ZeroesInput = torch.zeros(4, 3, 3, 3)

    def test_firstTemporalDerivative(self):
        # Dimension Check
        assert firstTemporalDerivative(self.ZeroesInput).size() == torch.Size([4, 3, 3, 3])
        assert firstTemporalDerivative(self.OnesInput).size() == torch.Size([4, 3, 3, 3])
        # Output Check
        # Should Output 4 * 3 * 3 * 3 tensor of zeroes
        assert firstTemporalDerivative(self.OnesInput).numpy().all() == 0.0
        assert firstTemporalDerivative(self.ZeroesInput).numpy().all() == 0.0

    def test_secondTemporalDerivative(self):
        # Dimension Check
        assert secondTemporalDerivative(self.OnesInput).size() == torch.Size([4, 3, 3, 3])
        assert secondTemporalDerivative(self.ZeroesInput).size() == torch.Size([4, 3, 3, 3])
        # Output Check
        # Should Output 4 * 3 * 3 * 3 tensor of zeroes
        assert secondTemporalDerivative(self.ZeroesInput).numpy().all() == 0.0
        assert secondTemporalDerivative(self.OnesInput).numpy().all() == True

# $ pytest -k TestRegularization

# Regularization Tests
class TestRegularization:
    OnesInput = torch.ones(4, 3, 3, 3)
    ZeroesInput = torch.zeros(4, 3, 3, 3)

    def test_thicknessRegularization(self):
        # Dimension Check
        assert thicknessRegularization(self.OnesInput, 1).size() == torch.Size([])
        assert thicknessRegularization(self.ZeroesInput, 1).size() == torch.Size([])
        # Output Check
        # Float Precision Error
        # This was the quick fix
        assert 35.99999 < thicknessRegularization(self.OnesInput, 1).item() < 36.0000
        assert thicknessRegularization(self.ZeroesInput, 1).item() == 0.0

    def test_roughnessRegularizaiton(self):
        # Dimension Check
        assert roughnessRegularization(self.OnesInput, 1).size() == torch.Size([])
        assert roughnessRegularization(self.ZeroesInput, 1).size() == torch.Size([])
        # Output Check
        assert roughnessRegularization(self.ZeroesInput, 1).item() == 0.0
        assert roughnessRegularization(self.OnesInput, 1).item() == 144.0

# $ pytest -k TestAdversarialLoss

class TestAdversarialLoss:
    # Labels dimension: same first dimension as predictions
    # second dimension n*1
    # each entry is an entry between 1 and x
    # Predictions
    Pred = torch.eye(10)
    Label = torch.arange(10)

    def test_adversarialLoss(self):
        # Output Shape to determine correct output
        print(self.Label.shape)
        loss = adversarialLoss(self.Pred, self.Label, 1)
        print(loss)
        assert torch.norm(loss, 1).item() == 20.

# $ pytest -k TestObjectiveFunc

class TestObjectiveFunc:
    # Random Seed
    torch.manual_seed(0)
    random.seed(0)

    # zeroes = torch.zeros(10, 10)
    # ones = torch.ones(10, 10)
    predictions = torch.rand(10, 10, requires_grad=True)
    labels = torch.randint(low=0, high=10, size=(10,))
    X = torch.rand(4, 3, 3, 3, requires_grad=True)

    # objectiveFunc.backward() creates None

    def test_objectiveFunc(self):
        loss = objectiveFunc(self.predictions, self.labels, self.X, 0.1, 1, 1, 0.5)

        print("\nLoss: %20.15f" % loss.item())
        # Ensure X is created correclty
        assert self.X.shape == torch.Size([4, 3, 3, 3])

        # Dimension Check
        assert loss.size() == torch.Size([])

        # Check if gradient and backward exist
        assert hasattr(loss, 'grad_fn')

        # Check if gradient and backward are callable
        assert callable(loss.grad_fn)

        # check backward
        loss.backward()

        # Confirm the gradient
        assert loss.grad_fn != None

        # Confirm the Output
        # Float Precision Issues This was the quick fix
        assert 1.7400<loss.item()<1.7401



