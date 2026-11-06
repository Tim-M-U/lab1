"""Real-data check: the perceptron on the iris dataset (setosa vs versicolor)."""

import numpy as np
from sklearn.datasets import load_iris

from perceptron_lab.model import FloatArray, IntArray

SETOSA = 0
VERSICOLOR = 1
DEFAULT_TEST_SIZE = 0.3


def load_iris_binary() -> tuple[FloatArray, IntArray]:
    """Load iris and keep only setosa (label 0) and versicolor (label 1).

    These two species are linearly separable, so a perceptron can tell them apart.

    Returns:
        Feature matrix of shape (100, 4) and labels of shape (100,).
    """
    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)
    mask = (y == SETOSA) | (y == VERSICOLOR)
    return X[mask], y[mask]


def train_test_split(
    X: FloatArray, y: IntArray, test_size: float = DEFAULT_TEST_SIZE, seed: int | None = None
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Shuffle the samples and split them into a training and a test part.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Labels of shape (n_samples,).
        test_size: Fraction of samples that go to the test part, between 0 and 1.
        seed: Seed for shuffling; the same seed gives the same split.

    Returns:
        X_train, X_test, y_train, y_test.

    Raises:
        ValueError: If test_size is not between 0 and 1 or a part would be empty.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    n_test = round(len(X) * test_size)
    if n_test == 0 or n_test == len(X):
        raise ValueError("not enough samples to split into train and test parts")

    order = np.random.default_rng(seed).permutation(len(X))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
