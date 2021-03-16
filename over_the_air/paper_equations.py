from torch import nn
from torch.nn import functional as F
import torch

__all__ = [

]


def tensorNorm(X: torch.Tensor, p: int) -> torch.Tensor:
    """
    Tensor p-norm. Equation 5 from the the over-the-air paper.
    :param X: `torch.tensor`
        Input tensor. Can be any dimensions, but per the paper it should be a
        4-dimensional Tensor with dimensions
        (T consecutive frames, H rows, W columns, C color channels).
    :param p: `int`
        The norm p value
    :return: `torch.tensor`
        A 1-D torch tensor with the scalar value of the norm.
    """

    # Per the PyTorch docs, this has been DEPRECATED. Since ART's requirements
    # specify torch version 1.6.0, we need to use this function because the
    # newer `torch.linalg.norm` does not exist in 1.6.0
    return torch.norm(X, p)


def firstTemporalDerivative(X: torch.Tensor) -> torch.Tensor:
    """
    Equation 7 from the paper.
    :param X: `torch.tensor`
        Input tensor. Can be any dimensions, but per the paper it should be a
        4-dimensional Tensor with dimensions
        (T consecutive frames, H rows, W columns, C color channels).
    :return: `torch.Tensor`
        The first order temporal derivative with dimensions
        (T consecutive frames, H rows, W columns, C color channels).
    """
    # Use dims to ensure that it is only shifted on the first dimensions. Per the paper,
    # we roll x_1,...,x_T in X. Since T is the first dimension of X, we use dim=0.
    return torch.roll(X, 1, dims=0) - torch.roll(X, 0, dims=0)


def secondTemporalDerivative(X: torch.Tensor) -> torch.Tensor:
    """
    Equation 8 from the paper. Defined as:
        Roll(X,-1) - 2*Roll(X, 0) + Roll(X,1)
    :param X: `torch.tensor`
        Input tensor. Can be any dimensions, but per the paper it should be a
        4-dimensional Tensor with dimensions
        (T consecutive frames, H rows, W columns, C color channels).
    :return: `torch.Tensor`
        The first order temporal derivative with dimensions
        (T consecutive frames, H rows, W columns, C color channels).
    """
    # Use dims to ensure that it is only shifted on the first dimensions. Per the paper,
    # we roll x_1,...,x_T in X. Since T is the first dimension of X, we use dim=0.
    return torch.roll(X, -1, dims=0) - 2 * torch.roll(X, 0, dims=0) - torch.roll(X, 1, dims=0)


# TODO: Delete this, and replace with full unittests
def testingMain():
    torch.manual_seed(1999)
    X = torch.rand(4, 3, 3, 3, requires_grad=True)

    # Sanity check
    assert X.shape == torch.Size([4, 3, 3, 3])

    dX = firstTemporalDerivative(X)
    d2X = secondTemporalDerivative(X)

    # Sanity check part 100000000
    assert dX.shape == torch.Size([4, 3, 3, 3])
    assert d2X.shape == torch.Size([4, 3, 3, 3])

    print("?")


if __name__ == '__main__':
    testingMain()
