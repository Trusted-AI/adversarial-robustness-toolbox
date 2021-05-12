# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import random

import torch

from art.attacks.evasion.over_the_air_flickering import OverTheAirFlickeringPyTorch
from tests.utils import TestBase


class TestOverTheAirFlickeringPyTorch(TestBase):
    OnesInput = torch.ones(4, 3, 3, 3)
    ZeroesInput = torch.zeros(4, 3, 3, 3)

    torch.manual_seed(0)
    random.seed(0)
    predictions = torch.rand(10, 10, requires_grad=True)
    labels = torch.randint(low=0, high=10, size=(10,))
    X = torch.rand(4, 3, 3, 3, requires_grad=True)

    def test_firstTemporalDerivative(self):
        assert OverTheAirFlickeringPyTorch._first_temporal_derivative(self.ZeroesInput).size() == torch.Size(
            [4, 3, 3, 3]
        )
        assert OverTheAirFlickeringPyTorch._first_temporal_derivative(self.OnesInput).size() == torch.Size([4, 3, 3, 3])
        assert OverTheAirFlickeringPyTorch._first_temporal_derivative(self.ZeroesInput).numpy().all() == 0.0
        assert OverTheAirFlickeringPyTorch._first_temporal_derivative(self.OnesInput).numpy().all() == 0.0

    def test_secondTemporalDerivative(self):
        assert OverTheAirFlickeringPyTorch._second_temporal_derivative(self.ZeroesInput).size() == torch.Size(
            [4, 3, 3, 3]
        )
        assert OverTheAirFlickeringPyTorch._second_temporal_derivative(self.OnesInput).size() == torch.Size(
            [4, 3, 3, 3]
        )
        assert OverTheAirFlickeringPyTorch._second_temporal_derivative(self.ZeroesInput).numpy().all() == 0.0
        assert OverTheAirFlickeringPyTorch._second_temporal_derivative(self.OnesInput).numpy().all()

    def test_thickness_regularization(self):
        assert OverTheAirFlickeringPyTorch._thickness_regularization(self.ZeroesInput, 1).size() == torch.Size([])
        assert OverTheAirFlickeringPyTorch._thickness_regularization(self.OnesInput, 1).size() == torch.Size([])
        # Float Precision Error
        assert OverTheAirFlickeringPyTorch._thickness_regularization(self.ZeroesInput, 1).item() == 0.0
        assert 35.99999 < OverTheAirFlickeringPyTorch._thickness_regularization(self.OnesInput, 1).item() < 36.00000

    """
    def test_roughness_regularization(self):
        assert OverTheAirFlickeringPyTorch._roughness_regularization(OverTheAirFlickeringPyTorch,
               self.ZeroesInput, 1).size() == torch.Size([])
        assert OverTheAirFlickeringPyTorch._roughness_regularization(OverTheAirFlickeringPyTorch,
               self.OnesInput, 1).size() == torch.Size([])
        assert OverTheAirFlickeringPyTorch._roughness_regularization(OverTheAirFlickeringPyTorch,
               self.ZeroesInput, 1).item() == 0.0
        assert OverTheAirFlickeringPyTorch._roughness_regularization(OverTheAirFlickeringPyTorch,
               self.OnesInput, 1).item() == 144.0

    def test_objective(self):
        loss = OverTheAirFlickeringPyTorch._objective(OverTheAirFlickeringPyTorch, self.predictions,
               self.labels, self.X)

        print("\nLoss: %20.15f" % loss.item())
        # Ensure X is created correclty
        assert self.X.shape == torch.Size([4, 3, 3, 3])

        # Dimension Check
        assert loss.size() == torch.Size([])

        # Check if gradient and backward exist
        assert hasattr(loss, "grad_fn")

        # Check if gradient and backward are callable
        assert callable(loss.grad_fn)

        # check backward
        loss.backward()

        # Confirm the gradient
        assert loss.grad_fn != None

        # Confirm the Output
        # Float Precision Issues This was the quick fix
        assert 1.7400 < loss.item() < 1.7401



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
        assert torch.norm(loss, 1).item() == 20.0


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
        assert hasattr(loss, "grad_fn")

        # Check if gradient and backward are callable
        assert callable(loss.grad_fn)

        # check backward
        loss.backward()

        # Confirm the gradient
        assert loss.grad_fn != None

        # Confirm the Output
        # Float Precision Issues This was the quick fix
        assert 1.7400 < loss.item() < 1.7401

"""
