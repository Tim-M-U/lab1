"""Command-line interface: train the perceptron and save the plots."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from perceptron_lab.data import make_blobs
from perceptron_lab.iris import load_iris_binary, train_test_split
from perceptron_lab.model import Perceptron
from perceptron_lab.plots import plot_decision_boundary, plot_errors

DEFAULT_PLOTS_DIR = Path("plots")


def positive_int(value: str) -> int:
    """Parse a positive integer argument for argparse."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {number}")
    return number


def positive_float(value: str) -> float:
    """Parse a positive float argument for argparse."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="perceptron",
        description="Train a single-layer perceptron and plot the results.",
    )
    parser.add_argument(
        "--samples", type=positive_int, default=200, help="number of synthetic points"
    )
    parser.add_argument("--lr", type=positive_float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=positive_int, default=50, help="maximum epochs")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="where to save the plots"
    )
    parser.add_argument(
        "--iris", action="store_true", help="check on iris (setosa vs versicolor) instead"
    )
    return parser


def run_synthetic(args: argparse.Namespace) -> None:
    """Train on two generated clouds of points and save both plots."""
    X, y = make_blobs(args.samples, seed=args.seed)
    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X, y)

    print(f"Synthetic data: {args.samples} points, 2 features")
    print(f"Epochs used:    {len(errors)} of {args.epochs}")
    print(f"Accuracy:       {model.score(X, y):.3f}")

    boundary_path = args.plots_dir / "decision_boundary.png"
    errors_path = args.plots_dir / "errors.png"
    plot_decision_boundary(model, X, y, boundary_path, "Perceptron on synthetic data")
    plot_errors(errors, errors_path, "Errors per epoch (synthetic data)")
    print(f"Plots saved:    {boundary_path}, {errors_path}")


def run_iris(args: argparse.Namespace) -> None:
    """Train on iris (setosa vs versicolor) and report train and test accuracy."""
    X, y = load_iris_binary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=args.seed)
    model = Perceptron(n_features=X.shape[1], learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X_train, y_train)

    print(f"Iris: setosa vs versicolor, {len(X_train)} train / {len(X_test)} test samples")
    print(f"Epochs used:    {len(errors)} of {args.epochs}")
    print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")

    errors_path = args.plots_dir / "iris_errors.png"
    plot_errors(errors, errors_path, "Errors per epoch (iris)")
    print(f"Plot saved:     {errors_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program; return the process exit code.

    Args:
        argv: Command-line arguments without the program name; None means sys.argv.

    Returns:
        0 on success, 1 if an error occurred.
    """
    args = build_parser().parse_args(argv)
    try:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        if args.iris:
            run_iris(args)
        else:
            run_synthetic(args)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
