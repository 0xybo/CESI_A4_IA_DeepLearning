from __future__ import annotations
from abc import ABC, abstractmethod
from ..neural_network import NeuralNetwork

# class Callback {
#     + NeuralNetwork neural_network

#     + None set_neural_network(NeuralNetwork neural_network)
#     + None on_batch_begin(batch: int)
#     + None on_batch_end(batch: int)
#     + None on_epoch_begin(epoch: int)
#     + None on_epoch_end(epoch: int)
#     + None on_train_batch_begin(batch: int)
#     + None on_train_batch_end(batch: int)
#     + None on_test_batch_begin(batch: int)
#     + None on_test_batch_end(batch: int)
#     + None on_predict_batch_begin(batch: int)
#     + None on_predict_batch_end(batch: int)
#     + None on_train_begin()
#     + None on_train_end()
#     + None on_test_begin()
#     + None on_test_end()
#     + None on_predict_begin()
#     + None on_predict_end()
# }

class Callback(ABC):
    neural_network: NeuralNetwork

    @abstractmethod
    def set_neural_network(self, neural_network: NeuralNetwork) -> None: ...
    @abstractmethod
    def on_batch_begin(self, batch: int) -> None: ...
    @abstractmethod
    def on_batch_end(self, batch: int) -> None: ...
    @abstractmethod
    def on_epoch_begin(self, epoch: int) -> None: ...
    @abstractmethod
    def on_epoch_end(self, epoch: int) -> None: ...
    @abstractmethod
    def on_train_batch_begin(self, batch: int) -> None: ...
    @abstractmethod
    def on_train_batch_end(self, batch: int) -> None: ...
    @abstractmethod
    def on_test_batch_begin(self, batch: int) -> None: ...
    @abstractmethod
    def on_test_batch_end(self, batch: int) -> None: ...
    @abstractmethod
    def on_predict_batch_begin(self, batch: int) -> None: ...
    @abstractmethod
    def on_predict_batch_end(self, batch: int) -> None: ...
    @abstractmethod
    def on_train_begin(self) -> None: ...
    @abstractmethod
    def on_train_end(self) -> None: ...
    @abstractmethod
    def on_test_begin(self) -> None: ...
    @abstractmethod
    def on_test_end(self) -> None: ...
    @abstractmethod
    def on_predict_begin(self) -> None: ...
    @abstractmethod
    def on_predict_end(self) -> None: ...