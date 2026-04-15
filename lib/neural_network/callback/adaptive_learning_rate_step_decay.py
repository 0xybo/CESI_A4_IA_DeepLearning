"""
Module for implementing a step decay learning rate scheduler as a callback for the neural
network training process.

The StepDecay callback reduces the learning rate by a specified factor every fixed number
of epochs. This can help the model converge more quickly and avoid overshooting minima in the
loss landscape.
"""

from .base import Callback  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level


class StepDecay(Callback):
    """
    Adaptive learning rate callback that implements step decay scheduling.

    Parameters:
    - initial_lr (float): Initial learning rate at the start of training.
    - drop_factor (float): Factor by which to reduce the learning rate at each step (e.g., 0.5 for
      halving).
    - epochs_per_drop (int): Number of epochs to wait before applying the drop factor.
    - min_lr (float): Minimum learning rate to prevent it from becoming too small.
    """

    initial_lr: float
    drop_factor: float
    epochs_per_drop: int
    min_lr: float

    def __init__(
        self,
        initial_lr: float,
        drop_factor: float = 0.5,
        epochs_per_drop: int = 10,
        min_lr: float = 1e-7,
    ):
        """
        Step Decay Scheduler.

        Parameters:
        - initial_lr: The LR at epoch 0.
        - drop_factor: Multiply LR by this factor every step (e.g., 0.5 drops it by 50%).
        - epochs_per_drop: How many epochs to wait before dropping the LR.
        - min_lr: The floor for the learning rate.
        """
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.epochs_per_drop = epochs_per_drop
        self.min_lr = min_lr

    def on_train_begin(self, neural_network: NeuralNetwork) -> None:
        """
        Called at the beginning of training. Resets the learning rate to the initial value.
        """
        neural_network.learning_rate = self.initial_lr

    def on_epoch_end(self, neural_network: NeuralNetwork, epoch: int) -> None:
        """
        Called at the end of each epoch. Calculates LR based on the current epoch number.
        """
        # Formula: initial_lr * (drop_factor ^ floor(epoch / epochs_per_drop))
        exponent = epoch // self.epochs_per_drop
        new_lr = self.initial_lr * (self.drop_factor**exponent)

        # Ensure we don't go below the floor
        neural_network.learning_rate = max(new_lr, self.min_lr)
