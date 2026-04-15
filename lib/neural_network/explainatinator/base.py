"""
Definition of the Explainatinator class, which provides methods for explaining the 
    predictions of a neural network model using various techniques such as LIME and SHAP.
"""
from abc import abstractmethod
import numpy as np
from ..neural_network import NeuralNetwork

class Explainatinator:
    """
    Class explainating the predictions of a neural network model using various 
        techniques such as LIME and SHAP.
    """
    model: NeuralNetwork

    def __init__(self, model: NeuralNetwork) -> None:
        self.model = model

    @abstractmethod
    def explain(self, x: np.ndarray) -> np.ndarray: ...
