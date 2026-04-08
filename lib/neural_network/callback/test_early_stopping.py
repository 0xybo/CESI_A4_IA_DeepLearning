"""
Test the EarlyStopping callback to ensure it correctly stops training when the validation loss does
not improve for a specified number of epochs (patience).
"""

import numpy as np
from .early_stopping import EarlyStopping
from ..neural_network import NeuralNetwork
from ..loss.binary_cross_entropy import BinaryCrossEntropy


class NeuralNetworkPlaceholder(NeuralNetwork):
    """
    Placeholder Neural Network for testing the EarlyStopping callback.
    This class simulates a neural network with a predefined history of training losses and
    predictions.
    """

    def __init__(self):
        super().__init__(layers=[], loss=BinaryCrossEntropy())
        self.history = [
            {
                "loss": 0.5,
                "val_loss": 0.6,
                "y_pred": np.array([[0.5], [0.5]]),
                "x_train": np.array([[0], [1]]),
                "y_train": np.array([[0], [1]]),
            }
        ]


def test_early_stopping():
    """
    Test the EarlyStopping callback by simulating a training process with a placeholder neural
    network.
    The test checks if the callback correctly stops training after the validation loss does not
    improve for a specified number of epochs (patience).
    """

    neural_network_placeholder = NeuralNetworkPlaceholder()
    callback = EarlyStopping(patience=3)

    assert callback.patience == 3
    assert callback.best_loss == float("inf")
    assert callback.counter == 0

    neural_network_placeholder.fiting = True

    for epoch in range(10):
        neural_network_placeholder.history.append(
            {
                "loss": 0.5 / (epoch + 1),
                "val_loss": 0.6 / (epoch + 1) if epoch < 5 else 0.6,
                "y_pred": neural_network_placeholder.predicts(
                    neural_network_placeholder.history[0]["x_train"]
                ),
                "x_train": neural_network_placeholder.history[0]["x_train"],
                "y_train": neural_network_placeholder.history[0]["y_train"],
            }
        )
        print(
            (
                f"Epoch {epoch + 1}:"
                f"val_loss = {neural_network_placeholder.history[-1]['val_loss']:.4f},"
                f"best_loss = {callback.best_loss:.4f}, counter = {callback.counter}"
            )
        )

        callback.on_epoch_end(neural_network_placeholder, epoch)

        if not neural_network_placeholder.fiting:
            print(f"Stopped at epoch {epoch + 1}")
            break


if __name__ == "__main__":
    test_early_stopping()
