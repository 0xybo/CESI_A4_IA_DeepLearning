from __future__ import annotations
import numpy as np
from typing import List, Optional
from .layer import Layer
from .callback.base import Callback

# class NeuralNetwork {
#     + Layer[] layers
#     + LossFunction loss
#     + Callback[] callbacks
#     + bool fiting
#     + Dataframe history - loss, validation
#     + ndarray x_train
#     + ndarray y_train
#     + int epochs
#     + int epoch
#     + int batch_size
#     + int validation_split

#     + None __init__(Layer[] layers)
#     + None addLayer(Layer layer)
#     + None addCallback(Callback callback)
#     + ndarray fit(ndarray x_train, ndarray y_train, int epochs, int batch_size, int validation_split, int learning_rate)
#     + float predict(ndarray x)
# }

class NeuralNetwork:
    def __init__(self, layers: List[Layer]) -> None: ...
    def addLayer(self, layer: Layer) -> None: ...
    def addCallback(self, callback: Callback) -> None: ...
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, validation_split: float, learning_rate: float) -> None: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...