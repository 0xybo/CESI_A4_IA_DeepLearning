from __future__ import annotations
from abc import ABC, abstractmethod

# class ActivationFunction {
#     + float compute(float x)
#     + float derivative(float x)
# }

class ActivationFunction(ABC):
    @abstractmethod
    def compute(self, x: float) -> float: ...
    @abstractmethod
    def derivative(self, x: float) -> float: ...