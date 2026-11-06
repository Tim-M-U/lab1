from pathlib import Path

import numpy as np
import pytest

from perceptron_lab import Perceptron
from perceptron_lab.cli import main
from perceptron_lab.iris import load_iris_binary, train_test_split


def test_load_iris_binary_keeps_two_classes() -> None:
    X, y = load_iris_binary()

    assert X.shape == (100, 4)
    assert set(np.unique(y)) == {0, 1}


def test_train_test_split_sizes_and_no_overlap() -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.array([0, 1] * 5, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=0)

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3
    all_rows = {tuple(row) for row in np.vstack([X_train, X_test])}
    assert all_rows == {tuple(row) for row in X}


def test_perceptron_separates_iris_on_test_data() -> None:
    X, y = load_iris_binary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)
    model = Perceptron(n_features=4)

    model.train(X_train, y_train)

    assert model.score(X_test, y_test) == 1.0


def test_cli_synthetic_run_saves_plots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--samples", "50", "--epochs", "20", "--plots-dir", str(tmp_path)])

    assert exit_code == 0
    assert "Accuracy:       1.000" in capsys.readouterr().out
    assert (tmp_path / "decision_boundary.png").exists()
    assert (tmp_path / "errors.png").exists()


def test_cli_iris_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--iris", "--plots-dir", str(tmp_path)])

    assert exit_code == 0
    assert "Test accuracy:" in capsys.readouterr().out
    assert (tmp_path / "iris_errors.png").exists()


def test_cli_rejects_negative_learning_rate(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--lr", "-1"])

    assert exit_info.value.code == 2
    assert "expected a positive number" in capsys.readouterr().err
