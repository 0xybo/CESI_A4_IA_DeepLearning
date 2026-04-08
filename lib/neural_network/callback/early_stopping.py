from .base import Callback
from ..neural_network import NeuralNetwork


class EarlyStopping(Callback):
    patience: int
    best_loss: float
    counter: int

    def __init__(self, patience: int = 5) -> None:
        super().__init__()
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, neural_network: NeuralNetwork, epoch: int) -> None:
        current_loss = neural_network.history[-1]["val_loss"]

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            neural_network.fiting = False
