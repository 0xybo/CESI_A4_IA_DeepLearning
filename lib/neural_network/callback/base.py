from __future__ import annotations
from abc import ABC, abstractmethod
from ..neural_network import NeuralNetwork

class Callback(ABC):
    neural_network: NeuralNetwork

    def set_neural_network(self, neural_network: NeuralNetwork) -> None: 
        self.neural_network = neural_network

    @abstractmethod
    def on_epoch_end(self, epoch: int) -> None: ...