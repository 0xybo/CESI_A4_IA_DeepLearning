import numpy as np
from .mean_squared_error import MeanSquaredError

def test_mean_squared_error_compute():
    loss = MeanSquaredError()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    expected_loss = np.mean((y_true - y_pred) ** 2)
    loss_value = loss.compute(y_true, y_pred)
    assert np.isclose(
        loss_value, expected_loss
    ), f"Mean Squared Error loss calculation is incorrect. Got: {loss_value} instead of {expected_loss}"

def test_mean_squared_error_derivative():
    loss = MeanSquaredError()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    expected_derivative = 2 * (y_pred - y_true) / y_true.shape[0]
    loss_derivative = loss.derivative(y_true, y_pred)
    assert np.allclose(
        loss_derivative, expected_derivative
    ), f"Mean Squared Error derivative calculation is incorrect. Got: {loss_derivative} instead of {expected_derivative}"

if __name__ == "__main__":
    test_mean_squared_error_compute()
    test_mean_squared_error_derivative()

    print("All tests passed!")