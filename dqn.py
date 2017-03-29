import itertools
import numpy as np

from typing import Callable

from skimage.transform import pyramid_reduce

np.random.seed(1)


def featurize(X: np.ndarray, target=(84, 84, 4)) -> np.ndarray:
    """Featurize the provided data.

    Simple image compression, for now.
    """
    return pyramid_reduce(X, scale=X.shape[0]/target[0])


def sigmoid(x, derivative=False):
    """Sigmoid activation function."""
    if derivative:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def learn(X: np.array, y: np.array, d: int,
          stopping_criterion: Callable[[int], bool]):
    """Train a two-layer neural network."""
    l0, w0 = X, np.random.random((d, 1))
    for t in itertools.count():

        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(t):
            break

        l1 = sigmoid(l0.dot(w0))
        l1_delta = (y - l1) * sigmoid(l1, derivative=True)
        w0 += l0.T.dot(l1_delta)
    return w0


