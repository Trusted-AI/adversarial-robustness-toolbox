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
    # Use dims to ensure that it is only shifted on the first dimensions. Per
    # the paper, we roll x_1,...,x_T in X. Since T is the first dimension of X,
    # we use dim=0.
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
    # Use dims to ensure that it is only shifted on the first dimensions.
    # Per the paper, we roll x_1,...,x_T in X. Since T is the first dimension of
    # X, we use dim=0.
    return torch.roll(X, -1, dims=0) - 2 * torch.roll(X, 0, dims=0) - torch.roll(X, 1, dims=0)


def objectiveFunc(predictions: torch.Tensor,
                  labels: torch.Tensor,
                  delta: torch.Tensor,
                  regularization_param: float,
                  beta_1: float,
                  beta_2: float,
                  m: float) -> torch.Tensor:
    """
    Equation (1): The objective function. Does NOT include the argmin nor constraints from
    equation (2).
    :param predictions:
    :param labels:
    :param delta:
    :param regularization_param:
    :param beta_1:
    :param beta_2:
    :return:
    """
    T = delta.shape[0]
    # The first summation from equation (1)
    regularization_term = regularization_param * (
            beta_1 * thicknessRegularization(delta, T)
            + beta_2 * roughnessRegularization(delta, T)
    )

    return regularization_term + torch.mean(adversarialLoss(predictions, labels, m))


def adversarialLoss(predictions: torch.Tensor, labels: torch.Tensor, m: float) -> torch.Tensor:
    """

    :param predictions: Logits?
    :param labels:
    :param m:
    :return:
    """
    # Number of samples x Number of Labels
    samples, n = predictions.shape
    pred_mask = torch.ones(samples, n).type(torch.bool)
    pred_mask[torch.arange(end=samples), labels[:]] = False

    # Equation (4) from the paper:
    #   You need the `==` or else pytorch throws a fit.
    #
    #   predictions[pred_mask == False]:
    #       Get the logits for the true labeled class
    #
    #   torch.max(predictions[pred_mask == True].view(samples,m-1), dim=-1)[0]:
    #       Get the max logit for each row that is not the true class.
    l_m = (
            predictions[pred_mask == False]
            - torch.max(predictions[pred_mask == True].view(samples, n - 1), dim=-1)[0]
            + m
    )

    # Equation 3
    return torch.max(torch.zeros(labels.shape), torch.min(1 / m * torch.pow(l_m, 2), l_m))


# TODO: Also, get rid of the garbage I call most of these comments.
def thicknessRegularization(delta: torch.Tensor, T: int) -> torch.Tensor:
    """
    Thickness Function
    :param delta: `torch.Tensor`
        Delta parameter from the paper
    :param T: `int`

    :return: `torch.Tensor`
        The THICKness. Like oatmeal * oatmeal=oatmeal^2
    """
    return torch.pow(tensorNorm(delta, 2), 2) / (3 * T)


def roughnessRegularization(delta: torch.Tensor, T: int) -> torch.Tensor:
    """
    ROUGH AND ROWDY
    :param delta: `torch.Tensor`
        Delta parameter from the paper
    :param T:
    :return:
        Rough.
    """
    return 1 / (3 * T) * (
            torch.pow(tensorNorm(firstTemporalDerivative(delta), 2), 2)
            + torch.pow(tensorNorm(secondTemporalDerivative(delta), 2), 2)
    )


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

    preds = torch.rand(10, 10, requires_grad=True)
    tmp_labels = torch.randint(low=0, high=10, size=(10,))
    loss = objectiveFunc(preds, tmp_labels, X, .1, 1, 1, .5)
    loss.backward()
    print(f"Loss: {loss}")


if __name__ == '__main__':
    testingMain()
