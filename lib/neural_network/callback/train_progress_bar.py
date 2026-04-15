"""
Progress bar callback for neural network training.

Displays a progress bar during training with useful metrics like:
- Current epoch and total epochs
- Epoch progress with visual bar (via tqdm)
- Training and validation loss
- Time elapsed and estimated time remaining
"""

from __future__ import annotations
from tqdm.auto import tqdm
from .base import Callback  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level


class TrainProgressBar(Callback):
    """
    Callback to display a progress bar during training using tqdm.

    This callback shows:
    - Current epoch and total epochs
    - Epoch progress with visual bar (via tqdm)
    - Training and validation loss
    - Time elapsed and estimated time remaining

    Attributes:
    - pbar (tqdm | None): Current tqdm progress bar instance.
    """

    bar_position: int = 0
    pbar: tqdm | None

    def __init__(self, position: int = 0) -> None:
        super().__init__()
        self.pbar = None
        self.bar_position = position

    def on_train_begin(self, neural_network: NeuralNetwork) -> None:
        """
        Called at the beginning of training.

        Initializes a tqdm progress bar for the training process.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
        """
        self.pbar = tqdm(
            total=neural_network.epochs,
            desc=f"Training {neural_network.name}",
            unit="epoch",
            position=self.bar_position,
        )

    def on_train_cancel(self, neural_network: NeuralNetwork) -> None:
        """
        Called if training is cancelled.

        Closes the progress bar.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
        """
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def on_epoch_end(self, neural_network: NeuralNetwork, epoch: int) -> None:
        """
        Called at the end of each epoch during training.

        Updates the progress bar with current loss metrics.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
            epoch (int): The current epoch number (0-indexed).
        """
        if self.pbar is None:
            return

        # Get metrics from current history
        history_entry = neural_network.history[-1] if neural_network.history else None

        if history_entry is not None:
            loss = history_entry["loss"]
            val_loss = history_entry["val_loss"]
            metrics = {
                "loss": f"{loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "lr": f"{neural_network.learning_rate:.4e}"
            }
        else:
            metrics = {}

        self.pbar.update(1)
        self.pbar.set_postfix(metrics)

    def on_train_end(self, neural_network: NeuralNetwork) -> None:
        """
        Called at the end of training.

        Closes the progress bar.

        Args:
            neural_network (NeuralNetwork): The neural network that was trained.
        """
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def __str__(self) -> str:
        return "TrainProgressBar"

    def __repr__(self) -> str:
        return self.__str__()
