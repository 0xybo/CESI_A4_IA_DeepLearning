from .base import LossFunction
import numpy as np

class BinaryCrossEntropy(LossFunction):
    def compute(
        self,
        y_train: np.ndarray, 
        y_pred: np.ndarray
    ) -> float: 
        # Clip y_pred to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y_one_pred = np.clip(1 - y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(
            y_train * np.log(y_pred) + (1 - y_train) * np.log(y_one_pred)
        ) # pyright: ignore[reportReturnType]
    
    def derivative(
        self,
        y_train: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        # Clip y_pred to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y_one_pred = np.clip(1 - y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_train) / (y_pred * y_one_pred * y_train.shape[0])