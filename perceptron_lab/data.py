"""Synthetic dataset generation for the perceptron."""

import numpy as np

from perceptron_lab.model import FloatArray, IntArray

CLASS_0_CENTER = (-2.0, -2.0)
CLASS_1_CENTER = (2.0, 2.0)
DEFAULT_SPREAD = 0.6


def make_blobs(
    n_samples: int, spread: float = DEFAULT_SPREAD, seed: int | None = None
) -> tuple[FloatArray, IntArray]:
    """Generate two linearly separable clouds of 2D points.

    Half of the points are drawn around CLASS_0_CENTER (label 0), the rest
    around CLASS_1_CENTER (label 1), then the samples are shuffled.

    Args:
        n_samples: Total number of points, at least 2.
        spread: Standard deviation of each cloud, must be positive.
        seed: Seed for the random generator; the same seed gives the same data.

    Returns:
        Feature matrix of shape (n_samples, 2) and labels of shape (n_samples,).

    Raises:
        ValueError: If n_samples < 2 or spread is not positive.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    if spread <= 0:
        raise ValueError(f"spread must be positive, got {spread}")

    rng = np.random.default_rng(seed)
    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0

    X_class_0 = rng.normal(loc=CLASS_0_CENTER, scale=spread, size=(n_class_0, 2))
    X_class_1 = rng.normal(loc=CLASS_1_CENTER, scale=spread, size=(n_class_1, 2))
    X = np.vstack([X_class_0, X_class_1])
    y = np.concatenate([np.zeros(n_class_0, dtype=np.int64), np.ones(n_class_1, dtype=np.int64)])

    order = rng.permutation(n_samples)
    return X[order], y[order]
