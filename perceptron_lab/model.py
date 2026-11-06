"""Single-layer perceptron for binary classification."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule.

    Attributes:
        n_features: Number of input features.
        learning_rate: Step size of the weight update.
        epochs: Maximum number of passes over the training set.
        weights: Weight vector of shape (n_features,).
        bias: Bias term (the activation threshold with a minus sign).
        errors_: Number of misclassified samples on each training epoch.
    """

    def __init__(self, n_features: int, learning_rate: float = 0.1, epochs: int = 50) -> None:
        """Create an untrained perceptron with zero weights.

        Args:
            n_features: Number of input features, must be positive.
            learning_rate: Step size of the weight update, must be positive.
            epochs: Maximum number of passes over the training set, must be positive.

        Raises:
            ValueError: If any argument is not positive.
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: FloatArray = np.zeros(n_features, dtype=np.float64)
        self.bias: float = 0.0
        self.errors_: list[int] = []

    @staticmethod
    def activate(z: float) -> int:
        """Apply the step activation function.

        Args:
            z: Weighted sum of the inputs plus bias.

        Returns:
            1 if z >= 0, otherwise 0.
        """
        return 1 if z >= 0 else 0

    def predict(self, x: FloatArray) -> int:
        """Predict the class of a single sample.

        Args:
            x: Feature vector of shape (n_features,).

        Returns:
            Predicted class, 0 or 1.

        Raises:
            ValueError: If x has the wrong number of features or contains NaN/inf.
        """
        if x.shape != (self.n_features,):
            raise ValueError(f"expected {self.n_features} features, got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("features must be finite numbers, got NaN or inf")
        z = float(np.dot(self.weights, x)) + self.bias
        return self.activate(z)

    def _check_data(self, X: FloatArray, y: IntArray) -> None:
        """Validate a feature matrix and its labels.

        Raises:
            ValueError: If X is not (n_samples, n_features), contains NaN/inf,
                or y does not match X or holds labels other than 0 and 1.
        """
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (n_samples, {self.n_features}), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numbers, got NaN or inf")
        if y.shape != (X.shape[0],):
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"y must contain only labels 0 and 1, got {np.unique(y).tolist()}")

    def train(self, X: FloatArray, y: IntArray) -> list[int]:
        """Fit the weights with the perceptron learning rule.

        Stops early once an epoch passes without errors.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels (0 or 1) of shape (n_samples,).

        Returns:
            Number of misclassified samples on each epoch (also stored in errors_).

        Raises:
            ValueError: If X or y is invalid (see _check_data).
        """
        self._check_data(X, y)
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y, strict=True):
                error = int(target) - self.predict(xi)
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                break
        return self.errors_

    def score(self, X: FloatArray, y: IntArray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True class labels of shape (n_samples,).

        Returns:
            Fraction of correctly classified samples, from 0.0 to 1.0.

        Raises:
            ValueError: If the data is empty or shapes do not match the model.
        """
        self._check_data(X, y)
        if len(y) == 0:
            raise ValueError("cannot compute accuracy on an empty dataset")
        correct = sum(1 for xi, target in zip(X, y, strict=True) if self.predict(xi) == target)
        return correct / len(y)
