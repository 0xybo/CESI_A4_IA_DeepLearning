"""
Base class for callbacks in the neural network training process.

This module defines the `Callback` class, which serves as a base class for creating custom
callbacks that can be used during the training of a neural network. Callbacks are functions
for methods that are called at specific points during the training process, such as at the 
beginning or end of an epoch, batch, or the entire training process. They allow users to perform
custom actions, such as logging, early stopping, or visualizing training progress.
"""

from __future__ import annotations

class Callback:
    """
    Base class for callbacks in the neural network training process.
    """
    def on_epoch_begin(self, neural_network: "NeuralNetwork", epoch: int) -> None:  # type: ignore
        """
        Called at the beginning of each epoch during training.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - epoch (int): The current epoch number (0-indexed).
        """

    async def on_epoch_begin_async(
        self, neural_network: "NeuralNetwork", epoch: int  # type: ignore
    ) -> None:
        """
        Asynchronous version of on_epoch_begin.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - epoch (int): The current epoch number (0-indexed).
        """

    def on_epoch_end(self, neural_network: "NeuralNetwork", epoch: int) -> None:  # type: ignore
        """
        Called at the end of each epoch during training.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - epoch (int): The current epoch number (0-indexed).
        """

    async def on_epoch_end_async(
        self, neural_network: "NeuralNetwork", epoch: int  # type: ignore
    ) -> None:
        """
        Asynchronous version of on_epoch_end.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - epoch (int): The current epoch number (0-indexed).
        """

    def on_batch_begin(self, neural_network: "NeuralNetwork", batch: int) -> None:  # type: ignore
        """
        Called at the beginning of each batch during training.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - batch (int): The current batch number (0-indexed).
        """

    async def on_batch_begin_async(
        self, neural_network: "NeuralNetwork", batch: int  # type: ignore
    ) -> None:
        """
        Asynchronous version of on_batch_begin.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - batch (int): The current batch number (0-indexed).
        """

    def on_batch_end(self, neural_network: "NeuralNetwork", batch: int) -> None:  # type: ignore
        """
        Called at the end of each batch during training.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - batch (int): The current batch number (0-indexed).
        """

    async def on_batch_end_async(
        self, neural_network: "NeuralNetwork", batch: int  # type: ignore
    ) -> None:
        """
        Asynchronous version of on_batch_end.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        - batch (int): The current batch number (0-indexed).
        """

    def on_train_begin(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        """
        Called at the beginning of the training process.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        """

    async def on_train_begin_async(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        """
        Asynchronous version of on_train_begin.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        """

    def on_train_end(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        """
        Called at the end of the training process.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        """

    async def on_train_end_async(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        """
        Asynchronous version of on_train_end.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        """

    def on_train_cancel(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        """
        Called when the training process is canceled.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        """

    async def on_train_cancel_async(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        """
        Asynchronous version of on_train_cancel.

        Args:
        - neural_network (NeuralNetwork): The neural network being trained.
        """
