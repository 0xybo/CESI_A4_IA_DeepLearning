from .base import LossFunction
import numpy as np

class MeanAbsoluteError(LossFunction):
    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float: ...