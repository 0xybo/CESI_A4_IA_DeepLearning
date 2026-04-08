from __future__ import annotations
import numpy as np
from .activation.base import ActivationFunction

# class Layer {
#     + int neurons
#     + ndarray weights
#     + float dropout_rate
#     + ActivationFunction activation

#     + None __init__(int neurons, float dropout_rate, ActivationFunction activation)
#     + ndarray forward(ndarray inputs)
#     + ndarray backward(ndarray dz)
# }

class Layer:
    neurons: int
    weights: np.ndarray
    dropout_rate: float
    activation: ActivationFunction

    def __init__(self, neurons: int, dropout_rate: float, activation: ActivationFunction) -> None: ...
    def forward(self, inputs: np.ndarray) -> np.ndarray: ...
    def backward(self, dz: np.ndarray) -> np.ndarray: ...