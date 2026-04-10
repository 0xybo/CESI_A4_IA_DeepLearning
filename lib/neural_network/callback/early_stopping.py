"""
Module for the EarlyStopping callback in a neural network training process.
"""

from .base import Callback  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when validation loss does not improve for a specified
    number of epochs (patience).

    Early stopping is a regularization technique used to prevent overfitting in neural networks.
    It monitors the validation loss during training and stops the training process if the validation
    loss does not improve for a specified number of consecutive epochs (patience).

    Parameters:
    - patience (int): Number of epochs to wait for improvement before stopping training.
        Default is 5.
    Attributes:
    - best_loss (float): The best validation loss observed during training.
    - counter (int): Counter to track the number of epochs since the last improvement in
        validation loss.
    """

    patience: int
    best_loss: float
    counter: int

    def __init__(self, patience: int = 5) -> None:
        super().__init__()
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, neural_network: NeuralNetwork, epoch: int) -> None:
        """
        Called at the end of each epoch during training. Checks if the validation loss has improved
        and updates the best loss and counter accordingly. If the counter exceeds the patience, it
        stops the training process.
        """
        current_loss = neural_network.history[-1]["val_loss"]

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            # print(
            #     (
            #         f"Early stopping at epoch {epoch + 1},"
            #         f"best validation loss: {self.best_loss:.4f}"
            #     )
            # )
            neural_network.fiting = False

    def __str__(self) -> str:
        return f"EarlyStopping(patience={self.patience})"

    def __repr__(self) -> str:
        return self.__str__()
