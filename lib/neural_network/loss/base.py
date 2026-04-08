from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def compute(
        self, 
        y_train: np.ndarray, 
        y_pred: np.ndarray
    ) -> float: ...

    @abstractmethod
    def derivative(
        self,
        y_train: np.ndarray,
        y_pred: np.ndarray
    ) -> float: ...
