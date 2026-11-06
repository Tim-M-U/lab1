from pathlib import Path

import numpy as np
import pytest

from perceptron_lab import Perceptron
from perceptron_lab.data import make_blobs
from perceptron_lab.plots import plot_decision_boundary, plot_errors


def test_make_blobs_shapes_and_labels() -> None:
    X, y = make_blobs(n_samples=101, seed=0)

    assert X.shape == (101, 2)
    assert y.shape == (101,)
    assert set(np.unique(y)) == {0, 1}
    assert np.sum(y == 0) == 50
    assert np.sum(y == 1) == 51


def test_make_blobs_same_seed_same_data() -> None:
    X1, y1 = make_blobs(n_samples=50, seed=42)
    X2, y2 = make_blobs(n_samples=50, seed=42)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_make_blobs_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="n_samples must be at least 2"):
        make_blobs(n_samples=1)


def test_perceptron_separates_blobs() -> None:
    X, y = make_blobs(n_samples=200, seed=42)
    model = Perceptron(n_features=2)

    model.train(X, y)

    assert model.score(X, y) == 1.0


def test_plots_are_saved(tmp_path: Path) -> None:
    X, y = make_blobs(n_samples=40, seed=1)
    model = Perceptron(n_features=2)
    errors = model.train(X, y)
    boundary_path = tmp_path / "boundary.png"
    errors_path = tmp_path / "errors.png"

    plot_decision_boundary(model, X, y, boundary_path)
    plot_errors(errors, errors_path)

    assert boundary_path.stat().st_size > 0
    assert errors_path.stat().st_size > 0
