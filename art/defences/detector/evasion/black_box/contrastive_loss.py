import numpy as np


def contrastive_loss(
    e_i: np.ndarray,
    e_p: np.ndarray,
    e_j: np.ndarray,
    e_n: np.ndarray,
    m: float = np.sqrt(10),
) -> np.ndarray:
    """
    Contrastive loss function as proposed in paper:
    | https://arxiv.org/pdf/1907.05587.pdf
    | We consider two pairs of elements. Pair 1 consists of x_i, an element from the training set,
    | and x_p a “positive” element perceptually similar to x_i.
    | Pair 2 consists of a different training element x_j, along with a negative example x_n,
    | an element not perceptually similar to x_j. The contrastive loss for their encodings (e_i, e_p), (e_j, e_n)
    """

    positive_pair = np.linalg.norm(e_i - e_p, axis=-1) ** 2
    negative_pair = max(0, m ** 2 - np.linalg.norm(e_j - e_n) ** 2)

    return positive_pair + negative_pair
