"""Tests for the Perceptron model."""

import numpy as np
import pytest

from perceptron_lab import Perceptron
from perceptron_lab.model import FloatArray, IntArray

Dataset = tuple[FloatArray, IntArray]


@pytest.fixture
def and_data() -> Dataset:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    return X, y


def test_activate_threshold() -> None:
    assert Perceptron.activate(0.0) == 1
    assert Perceptron.activate(-1.0) == 0
    assert Perceptron.activate(5.0) == 1


def test_init_creates_zero_weights() -> None:
    model = Perceptron(n_features=3)
    assert model.weights.shape == (3,)
    assert np.all(model.weights == 0.0)
    assert model.bias == 0.0
    assert model.errors_ == []


def test_predict_with_known_weights() -> None:
    model = Perceptron(n_features=2)
    model.weights = np.array([1.0, 1.0], dtype=np.float64)
    model.bias = -1.5

    assert model.predict(np.array([1.0, 1.0], dtype=np.float64)) == 1
    assert model.predict(np.array([0.0, 0.0], dtype=np.float64)) == 0
    assert model.predict(np.array([1.0, 0.0], dtype=np.float64)) == 0


def test_train_separates_and_function(and_data: Dataset) -> None:
    X, y = and_data
    model = Perceptron(n_features=2)

    model.train(X, y)

    assert model.score(X, y) == 1.0


def test_train_stops_early_when_no_errors(and_data: Dataset) -> None:
    X, y = and_data
    model = Perceptron(n_features=2, epochs=100)

    errors = model.train(X, y)

    assert errors[-1] == 0
    assert len(errors) < 100
    assert errors == model.errors_


def test_train_cannot_separate_xor() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    model = Perceptron(n_features=2, epochs=100)

    errors = model.train(X, y)

    assert len(errors) == 100
    assert all(count > 0 for count in errors)
    assert model.score(X, y) < 1.0


@pytest.mark.parametrize(
    ("n_features", "learning_rate", "epochs", "message"),
    [
        (0, 0.1, 50, "n_features must be positive, got 0$"),
        (2, -1.0, 50, "learning_rate must be positive, got -1.0$"),
        (2, 0.1, 0, "epochs must be positive, got 0$"),
    ],
)
def test_invalid_arguments_raise(
    n_features: int, learning_rate: float, epochs: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Perceptron(n_features, learning_rate, epochs)


def test_predict_rejects_wrong_feature_count() -> None:
    model = Perceptron(n_features=2)

    with pytest.raises(ValueError, match="expected 2 features"):
        model.predict(np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_train_rejects_mismatched_labels(and_data: Dataset) -> None:
    X, y = and_data
    model = Perceptron(n_features=2)

    with pytest.raises(ValueError, match="y must have shape"):
        model.train(X, y[:3])


def test_train_rejects_labels_other_than_0_and_1(and_data: Dataset) -> None:
    X, _ = and_data
    y = np.array([-1, -1, -1, 1], dtype=np.int64)
    model = Perceptron(n_features=2)

    with pytest.raises(ValueError, match="only labels 0 and 1"):
        model.train(X, y)


def test_nan_features_are_rejected(and_data: Dataset) -> None:
    X, y = and_data
    X_with_nan = X.copy()
    X_with_nan[0, 0] = np.nan
    model = Perceptron(n_features=2)

    with pytest.raises(ValueError, match="finite"):
        model.train(X_with_nan, y)
    with pytest.raises(ValueError, match="finite"):
        model.predict(np.array([np.nan, 1.0], dtype=np.float64))
