"""Plots for perceptron results: decision boundary and errors per epoch."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from perceptron_lab.model import FloatArray, IntArray, Perceptron

CLASS_COLORS = ("tab:orange", "tab:blue")
MARGIN = 1.0
GRID_SIZE = 200


def plot_decision_boundary(
    model: Perceptron, X: FloatArray, y: IntArray, path: Path, title: str = "Decision boundary"
) -> None:
    """Save a scatter plot of two classes with the model's decision boundary.

    The plane is shaded by the predicted class and the line w1*x1 + w2*x2 + b = 0
    is drawn on top.

    Args:
        model: Trained perceptron with exactly 2 features.
        X: Feature matrix of shape (n_samples, 2).
        y: Class labels (0 or 1) of shape (n_samples,).
        path: Where to save the image (PNG).
        title: Plot title.

    Raises:
        ValueError: If the model or the data is not two-dimensional.
    """
    if model.n_features != 2 or X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("decision boundary can only be drawn for 2 features")

    x_min, x_max = X[:, 0].min() - MARGIN, X[:, 0].max() + MARGIN
    y_min, y_max = X[:, 1].min() - MARGIN, X[:, 1].max() + MARGIN

    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, GRID_SIZE), np.linspace(y_min, y_max, GRID_SIZE))
    zz = model.weights[0] * xx + model.weights[1] * yy + model.bias
    ax.contourf(xx, yy, zz >= 0, levels=[-0.5, 0.5, 1.5], colors=CLASS_COLORS, alpha=0.15)
    ax.contour(xx, yy, zz, levels=[0.0], colors="black", linewidths=2)

    for label, color in enumerate(CLASS_COLORS):
        mask = y == label
        ax.scatter(
            X[mask, 0], X[mask, 1], c=color, edgecolors="black", s=25, label=f"class {label}"
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_errors(errors: list[int], path: Path, title: str = "Errors per epoch") -> None:
    """Save a line plot of misclassified samples on each training epoch.

    Args:
        errors: Number of errors per epoch, as returned by Perceptron.train.
        path: Where to save the image (PNG).
        title: Plot title.

    Raises:
        ValueError: If errors is empty.
    """
    if not errors:
        raise ValueError("errors list is empty, train the model first")

    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    epochs = range(1, len(errors) + 1)
    ax.plot(epochs, errors, marker="o")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassified samples")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
