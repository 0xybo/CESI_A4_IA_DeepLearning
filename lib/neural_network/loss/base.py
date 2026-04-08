from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float: ...
    @abstractmethod
    def derivative(
        self,
        y_train: np.ndarray[np._AnyShapeT, np.dtype[np.float64]],
        y_pred: np.ndarray[np._AnyShapeT, np.dtype[np.float64]],
    ) -> float: ...
