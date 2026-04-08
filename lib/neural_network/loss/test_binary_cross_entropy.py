import numpy as np
from .binary_cross_entropy import BinaryCrossEntropy


def test_binary_cross_entropy_compute():
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
    assert np.isclose(
        loss_value, expected_loss
    ), f"Binary Cross Entropy loss calculation is incorrect. Got: {loss_value} instead of {expected_loss}"


def test_binary_cross_entropy_compute_with_zeros():
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
    assert np.isclose(
        loss_value, expected_loss
    ), f"Binary Cross Entropy loss calculation with zeros is incorrect. Got: {loss_value} instead of {expected_loss}"


def test_binary_cross_entropy_compute_with_ones():
    loss = BinaryCrossEntropy()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])  # This will cause log(1) issues
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
    assert np.isclose(
        loss_value, expected_loss
    ), f"Binary Cross Entropy loss calculation with ones is incorrect. Got: {loss_value} instead of {expected_loss}"


def test_binary_cross_entropy_compute_with_mismatched_shapes():
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
    assert np.allclose(
        loss.derivative(y_true, y_pred), expected_derivative
    ), "Binary Cross Entropy derivative calculation is incorrect."


if __name__ == "__main__":
    test_binary_cross_entropy_compute()
    test_binary_cross_entropy_compute_with_zeros()
    test_binary_cross_entropy_compute_with_ones()
    test_binary_cross_entropy_compute_with_mismatched_shapes()
    test_binary_cross_entropy_derivative()
    print("All tests passed!")
