"""
Progress bar callback for neural network training.

Displays a progress bar during training with useful metrics like:
- Current epoch and total epochs
- Batch progress with visual bar (via tqdm)
- Training and validation loss
- Time elapsed and estimated time remaining
"""

from __future__ import annotations
from tqdm import tqdm
from .base import Callback  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level


class EpochProgressBar(Callback):
    """
    Callback to display a progress bar during training using tqdm.

    This callback shows:
    - Epoch number and total epochs
    - Progress bar with batch progress
    - Metrics: loss, val_loss
    - Time elapsed and estimated time remaining

    Attributes:
    - pbar (tqdm | None): Current tqdm progress bar instance.
    """

    pbar: tqdm | None

    def __init__(self) -> None:
        super().__init__()
        self.pbar = None

    def on_train_begin(
        self,
        neural_network: NeuralNetwork,
    ) -> None:
        """
        Called at the beginning of training.

        Initializes a tqdm progress bar for the training process.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
        """
        self.total_batches = (
            neural_network.x_train.shape[0]
            - int(neural_network.x_train.shape[0] * neural_network.validation_split)
        ) // neural_network.batch_size + 1

        epoch_number = 0
        total_epochs = neural_network.epochs
        desc = f"Epoch {epoch_number}/{total_epochs} of {neural_network.name}"

        self.pbar = tqdm(
            total=self.total_batches,
            desc=desc,
            unit="batch",
            ncols=100,
        )

    def on_batch_end(self, neural_network: NeuralNetwork, batch: int) -> None:
        """
        Called at the end of each batch during training.

        Updates the progress bar with current loss metrics.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
            batch (int): The current batch number (0-indexed).
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
            }
        else:
            metrics = {}

        self.pbar.update(1)
        self.pbar.set_postfix(metrics)

    def on_epoch_end(self, neural_network: NeuralNetwork, epoch: int) -> None:
        """
        Called at the end of each epoch during training.

        Closes the progress bar and displays final metrics for the epoch.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
            epoch (int): The current epoch number (0-indexed).
        """
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

        if not neural_network.history:
            return

        history_entry = neural_network.history[-1]
        loss = history_entry["loss"]
        val_loss = history_entry["val_loss"]

        print(f"  loss: {loss:.4f} - val_loss: {val_loss:.4f}")

    def __str__(self) -> str:
        return "EpochProgressBar"

    def __repr__(self) -> str:
        return self.__str__()
