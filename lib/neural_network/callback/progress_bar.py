"""
Progress bar callback for neural network training.

Displays a Keras-like progress bar during training with useful metrics like:
- Current epoch and total epochs
- Batch progress with visual bar
- Training and validation accuracy
- Training and validation loss
- Time elapsed and estimated time per sample
"""

from __future__ import annotations
import time
from .base import Callback  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level


class ProgressBar(Callback):
    """
    Callback to display a Keras-like progress bar during training.

    This callback shows:
    - Epoch number and total epochs
    - Progress bar with the current batch and total batches
    - Metrics: accuracy, loss, val_accuracy, val_loss
    - Time elapsed and time per step

    Attributes:
    - epoch_start_time (float): Timestamp when the epoch started.
    - batch_start_time (float): Timestamp when the batch started.
    - total_batches (int): Total number of batches per epoch.
    """

    epoch_start_time: float
    batch_start_time: float
    total_batches: int

    def __init__(self) -> None:
        super().__init__()
        self.epoch_start_time = 0.0
        self.batch_start_time = 0.0
        self.total_batches = 0

    def __format_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """
        Create a Keras-like progress bar string.

        Args:
            current (int): Current progress.
            total (int): Total progress.
            width (int): Width of the progress bar.

        Returns:
            str: Formatted progress bar string.
        """
        if total == 0:
            return "━" * width

        filled = int((current / total) * width)
        empty = width - filled
        return "━" * filled + " " * empty

    def __format_time(self, seconds: float) -> str:
        """
        Format time in a human-readable format.

        Args:
            seconds (float): Time in seconds.

        Returns:
            str: Formatted time string.
        """
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        return f"{seconds:.1f}s"

    def on_epoch_begin(
        self,
        neural_network: NeuralNetwork,
        epoch: int,  # pylint: disable=unused-argument
    ) -> None:
        """
        Called at the beginning of each epoch.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
            epoch (int): The current epoch number (0-indexed).
        """
        self.epoch_start_time = time.time()
        self.total_batches = (
            neural_network.x_train.shape[0]
            - int(neural_network.x_train.shape[0] * neural_network.validation_split)
        ) // neural_network.batch_size + 1

    def on_batch_end(self, neural_network: NeuralNetwork, batch: int) -> None:
        """
        Called at the end of each batch during training.

        Displays a progress bar with current loss metrics.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
            batch (int): The current batch number (0-indexed).
        """
        current_batch = batch + 1
        epoch_number = neural_network.epoch + 1
        total_epochs = neural_network.epochs

        # Calculate time
        epoch_time = time.time() - self.epoch_start_time
        time_per_batch = epoch_time / current_batch

        # Get metrics from current history
        history_entry = neural_network.history[-1] if neural_network.history else None

        if history_entry is not None:
            loss = history_entry["loss"]
            val_loss = history_entry["val_loss"]
        else:
            loss = 0.0
            val_loss = 0.0

        # Format progress bar
        progress_bar = self.__format_progress_bar(current_batch, self.total_batches)
        time_str = self.__format_time(epoch_time)
        time_per_step = self.__format_time(time_per_batch)

        # Build output string (similar to Keras format)
        # Print epoch info once per epoch
        if current_batch == 1:
            print(f"Epoch {epoch_number}/{total_epochs}")

        output = (
            f"\r{current_batch}/{self.total_batches} {progress_bar} "
            f"{time_str} {time_per_step}/step - "
            f"loss: {loss:.4f} - val_loss: {val_loss:.4f}"
        )

        print(output, end="", flush=True)

    def on_epoch_end(self, neural_network: NeuralNetwork, epoch: int) -> None:
        """
        Called at the end of each epoch during training.

        Displays final metrics for the epoch.

        Args:
            neural_network (NeuralNetwork): The neural network being trained.
            epoch (int): The current epoch number (0-indexed).
        """
        if not neural_network.history:
            return

        history_entry = neural_network.history[-1]
        loss = history_entry["loss"]
        val_loss = history_entry["val_loss"]

        total_time = time.time() - self.epoch_start_time

        # Print final epoch summary
        print(
            f"\nEpoch {epoch + 1}/{neural_network.epochs} - "
            f"{self.__format_time(total_time)} - "
            f"loss: {loss:.4f} - "
            f"val_loss: {val_loss:.4f}\n"
        )

    def __str__(self) -> str:
        return "ProgressBar"

    def __repr__(self) -> str:
        return self.__str__()
