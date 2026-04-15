"""
Adaptive Learning Rate Callback

This callback adjusts the learning rate during training based on the validation loss.
The learning rate is increased at the beginning of training to speed up convergence and then
decreased when the validation loss plateaus to fine-tune the model.

The learning rate is updated as follows:
- If the validation loss improves, the learning rate is increased by a factor of 1.1 (up to a
    maximum of 0.1).
- If the validation loss does not improve for 3 consecutive epochs, the learning rate is
    decreased by a factor of 0.5 (down to a minimum of 1e-6).
- The learning rate is reset to the initial value at the start of training.
"""

from .base import Callback  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level


class ReduceOnPlateau(Callback):
    """
    Adaptive learning rate callback to adjust the learning rate during training based on validation
    loss.

    This callback increases the learning rate at the beginning of training to speed up convergence
    and decreases it when the validation loss plateaus to fine-tune the model.

    Parameters:
    - initial_lr (float): Initial learning rate. Default is 0.01.
    - increase_factor (float): Factor to increase the learning rate when validation loss improves.
        Default is 1.1.
    - decrease_factor (float): Factor to decrease the learning rate when validation loss does not
        improve. Default is 0.5.
    - patience (int): Number of epochs to wait for improvement before decreasing the learning
        rate. Default is 3.
    - min_lr (float): Minimum learning rate. Default is 1e-6.
    - max_lr (float): Maximum learning rate. Default is 0.1.
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-6,
        max_lr: float = 0.1,
        min_delta=1e-4,
    ) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_train_begin(self, neural_network: NeuralNetwork) -> None:
        """
        Called at the beginning of training. Resets the learning rate to the initial value and
        initializes the best loss and counter.
        """
        neural_network.learning_rate = self.initial_lr
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(
        self,
        neural_network: NeuralNetwork,
        epoch: int,  # pylint: disable=unused-argument
    ) -> None:
        """
        Called at the end of each epoch during training. Adjusts the learning rate based on the
        validation loss.
        """
        current_loss = neural_network.history[-1]["val_loss"]

        if current_loss < self.best_loss - self.min_delta:
            # Validation loss improved
            self.best_loss = current_loss
            self.counter = 0

            neural_network.learning_rate = min(
                neural_network.learning_rate * self.increase_factor, self.max_lr
            )
        else:
            # Validation loss did not improve
            self.counter += 1
            if self.counter >= self.patience:
                neural_network.learning_rate = max(
                    neural_network.learning_rate * self.decrease_factor, self.min_lr
                )
                self.counter = 0  # Reset counter after reducing learning rate

    def __str__(self) -> str:
        return (
            f"ReduceOnPlateau(initial_lr={self.initial_lr}, "
            f"increase_factor={self.increase_factor}, "
            f"decrease_factor={self.decrease_factor}, "
            f"patience={self.patience}, "
            f"min_lr={self.min_lr}, "
            f"max_lr={self.max_lr})"
        )
