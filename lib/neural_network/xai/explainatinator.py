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
    x_data: np.ndarray

    def __init__(self, model: NeuralNetwork, x_data: np.ndarray) -> None:
        self.model = model
        self.x_data = x_data

    @abstractmethod
    def explain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...
