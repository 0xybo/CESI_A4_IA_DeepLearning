from .base import LossFunction
import numpy as np

class BinaryCrossEntropy(LossFunction):
    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float: ...