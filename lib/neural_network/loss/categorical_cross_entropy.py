from .base import LossFunction
import numpy as np


class CategoricalCrossEntropy(LossFunction):
    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip y_pred to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(
            np.sum(y_train * np.log(y_pred), axis=1)
        )  # pyright: ignore[reportReturnType]

    def derivative(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip y_pred to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_train) / y_train.shape[0]
