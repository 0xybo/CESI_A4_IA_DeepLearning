"""
Test cases for the Categorical Cross Entropy loss function.
"""

import numpy as np
from .categorical_cross_entropy import CategoricalCrossEntropy


def test_categorical_cross_entropy_compute():
    """
    Test the compute method of the Categorical Cross Entropy loss function with a simple example.
    """
    loss = CategoricalCrossEntropy()
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    expected_loss = -np.mean(
        np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)), axis=1)
    )
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(loss_value, expected_loss), (
        "Categorical Cross Entropy loss calculation is incorrect."
        f"Got: {loss_value} instead of {expected_loss}"
    )


def test_categorical_cross_entropy_compute_with_zeros():
    """
    Test the compute method of the Categorical Cross Entropy loss function with predicted
    probabilities of 0, which should be clipped to prevent log(0) issues.
    """

    loss = CategoricalCrossEntropy()
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )  # This will cause log(0) issues
    expected_loss = -np.mean(
        np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)), axis=1)
    )
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(loss_value, expected_loss), (
        "Categorical Cross Entropy loss calculation with zeros is incorrect.",
        f"Got: {loss_value} instead of {expected_loss}",
    )


def test_categorical_cross_entropy_compute_with_ones():
    """
    Test the compute method of the Categorical Cross Entropy loss function with predicted
    probabilities of 1, which should be clipped to prevent log(1) issues.
    """
    loss = CategoricalCrossEntropy()
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )  # This will cause log(1) issues
    expected_loss = -np.mean(
        np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)), axis=1)
    )
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(loss_value, expected_loss), (
        "Categorical Cross Entropy loss calculation with ones is incorrect.",
        f"Got: {loss_value} instead of {expected_loss}",
    )


def test_categorical_cross_entropy_derivative():
    """
    Test the derivative method of the Categorical Cross Entropy loss function with a simple
    example.
    """
    loss = CategoricalCrossEntropy()
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    expected_derivative = (y_pred - y_true) / y_true.shape[0]
    assert np.allclose(
        loss.derivative(y_true, y_pred), expected_derivative
    ), "Categorical Cross Entropy derivative calculation is incorrect."


if __name__ == "__main__":
    test_categorical_cross_entropy_compute()
    test_categorical_cross_entropy_compute_with_zeros()
    test_categorical_cross_entropy_compute_with_ones()
    test_categorical_cross_entropy_derivative()

    print("All tests passed!")
