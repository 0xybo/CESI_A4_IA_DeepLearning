import numpy as np
from .mean_absolute_error import MeanAbsoluteError


def test_mean_absolute_error_compute():
    loss = MeanAbsoluteError()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    expected_loss = np.mean(np.abs(y_true - y_pred))
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(
        loss_value, expected_loss
    ), f"Mean Absolute Error loss calculation is incorrect. Got: {loss_value} instead of {expected_loss}"


def test_mean_absolute_error_derivative():
    loss = MeanAbsoluteError()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    # dMAE/dy_pred = { +1 if y_pred > y_true, -1 if y_pred < y_true
    expected_derivative = np.sign(y_pred - y_true) / y_true.shape[0]
    loss_derivative = loss.derivative(y_true, y_pred)
    assert np.allclose(
        loss_derivative, expected_derivative
    ), f"Mean Absolute Error derivative calculation is incorrect. Got: {loss_derivative} instead of {expected_derivative}"


if __name__ == "__main__":
    test_mean_absolute_error_compute()
    test_mean_absolute_error_derivative()

    print("All tests passed!")