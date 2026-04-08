from __future__ import annotations
import numpy as np
import pandas as pd
from .neural_network import NeuralNetwork
from typing import Optional, Tuple

# class Evaluation {
#     + NeuralNetwork neural_network
#     + ndarray x_validation
#     + ndarray y_validation
#     + tuple[int] _confusion_matrix

#     + None __init__(ndarray x_validation?, ndarray y_validation?)
#     + Dataframe validate(NeuralNetwork neural_network)
#     + float accuracy()
#     + float precision()
#     + float recall()
#     + float f1_score()
#     + float auc()
#     + None draw_roc()
#     + tuple[int] confusion_matrix()
# }

class Evaluation:
    neural_network: NeuralNetwork
    x_validation: np.ndarray
    y_validation: np.ndarray
    _confusion_matrix: Tuple[int, int, int, int]

    def __init__(self, x_validation: Optional[np.ndarray] = None, y_validation: Optional[np.ndarray] = None) -> None: ...
    def validate(self, neural_network: NeuralNetwork) -> pd.DataFrame: ...
    def accuracy(self, confusion_matrix: Optional[Tuple[int, int, int, int]] = None) -> float: ...
    def precision(self, confusion_matrix: Optional[Tuple[int, int, int, int]] = None) -> float: ...
    def recall(self, confusion_matrix: Optional[Tuple[int, int, int, int]] = None) -> float: ...
    def f1_score(self, confusion_matrix: Optional[Tuple[int, int, int, int]] = None) -> float: ...
    def auc(self) -> float: ...
    def draw_roc(self) -> None: ...
    def confusion_matrix(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]: ...