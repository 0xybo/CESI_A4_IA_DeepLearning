"""
Unit tests for the Binary Cross Entropy loss function.
"""

import numpy as np
from .binary_cross_entropy import BinaryCrossEntropy


def test_binary_cross_entropy_compute():
    """
    Test the compute method of the Binary Cross Entropy loss function with a simple example.
    """
    loss = BinaryCrossEntropy()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    expected_loss = (
        -(
            (1 * np.log(0.9) + (1 - 1) * np.log(1 - 0.9))
            + (0 * np.log(0.1) + (1 - 0) * np.log(1 - 0.1))
            + (1 * np.log(0.8) + (1 - 1) * np.log(1 - 0.8))
            + (0 * np.log(0.2) + (1 - 0) * np.log(1 - 0.2))
        )
        / 4
    )
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(loss_value, expected_loss), (
        "Binary Cross Entropy loss calculation is incorrect."
        f"Got: {loss_value} instead of {expected_loss}"
    )


def test_binary_cross_entropy_compute_with_zeros():
    """
    Test the compute method of the Binary Cross Entropy loss function with predicted probabilities
    of 0, which should be clipped to prevent log(0) issues.
    """
    loss = BinaryCrossEntropy()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0])  # This will cause log(0) issues
    expected_loss = (
        -(
            (1 * np.log(1e-15) + (1 - 1) * np.log(1 - 1e-15))
            + (0 * np.log(1e-15) + (1 - 0) * np.log(1 - 1e-15))
            + (1 * np.log(1e-15) + (1 - 1) * np.log(1 - 1e-15))
            + (0 * np.log(1e-15) + (1 - 0) * np.log(1 - 1e-15))
        )
        / 4
    )
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(loss_value, expected_loss), (
        "Binary Cross Entropy loss calculation with zeros is incorrect."
        f"Got: {loss_value} instead of {expected_loss}"
    )


def test_binary_cross_entropy_compute_with_ones():
    """
    Test the compute method of the Binary Cross Entropy loss function with predicted probabilities
    of 1, which should be clipped to prevent log(0) issues.
    """

    loss = BinaryCrossEntropy()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    expected_loss = (
        -(
            (1 * np.log(1.0) + (1 - 1) * np.log(1 - 1.0 + 1e-15))
            + (0 * np.log(1.0) + (1 - 0) * np.log(1 - 1.0 + 1e-15))
            + (1 * np.log(1.0) + (1 - 1) * np.log(1 - 1.0 + 1e-15))
            + (0 * np.log(1.0) + (1 - 0) * np.log(1 - 1.0 + 1e-15))
        )
        / 4
    )
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(loss_value, expected_loss), (
        "Binary Cross Entropy loss calculation with ones is incorrect."
        f"Got: {loss_value} instead of {expected_loss}"
    )


def test_binary_cross_entropy_compute_with_mismatched_shapes():
    """
    Test the compute method of the Binary Cross Entropy loss function with mismatched shapes for
    y_true and y_pred, which should raise an error.
    """
    loss = BinaryCrossEntropy()
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1])  # Mismatched shape
    try:
        loss.compute(y_true, y_pred)
        assert (
            False
        ), "Binary Cross Entropy should raise an error for mismatched shapes."
    except ValueError:
        pass  # Expected


def test_binary_cross_entropy_derivative():
    """
    Test the derivative method of the Binary Cross Entropy loss function with a simple example.
    """
    loss = BinaryCrossEntropy()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    expected_derivative = np.array(
        [
            (0.9 - 1) / (0.9 * 0.1 * 4),
            (0.1 - 0) / (0.1 * 0.9 * 4),
            (0.8 - 1) / (0.8 * 0.2 * 4),
            (0.2 - 0) / (0.2 * 0.8 * 4),
        ]
    )
    loss_derivative = loss.derivative(y_true, y_pred)
    assert np.allclose(loss_derivative, expected_derivative), (
        "Binary Cross Entropy derivative calculation is incorrect."
        f"Got: {loss_derivative} instead of {expected_derivative}"
    )


if __name__ == "__main__":
    test_binary_cross_entropy_compute()
    test_binary_cross_entropy_compute_with_zeros()
    test_binary_cross_entropy_compute_with_ones()
    test_binary_cross_entropy_compute_with_mismatched_shapes()
    test_binary_cross_entropy_derivative()
    print("All tests passed!")
