from .base import LossFunction
import numpy as np


class MeanAbsoluteError(LossFunction):
    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_train - y_pred))  # pyright: ignore[reportReturnType]

    def derivative(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sign(y_pred - y_train) / y_train.shape[0]
