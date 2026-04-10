"""
Class DrawRealTimeLoss: A callback to draw real-time loss and validation loss during training using
Matplotlib.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .base import Callback
from ..neural_network import NeuralNetwork


class DrawRealTimeLoss(Callback):  # pylint: disable=too-few-public-methods
    """
    Callback to draw real-time loss and validation loss during training.

    This callback uses Matplotlib to create a real-time plot of the training loss and validation
    loss after each epoch. It updates the plot with the latest loss values from the neural
    network's history.

    Attributes:
    - figure (Figure): The Matplotlib figure object for the plot.
    - axes (Axes): The Matplotlib axes object for the plot.
    - loss_line (Line2D): The line object for the training loss.
    - val_loss_line (Line2D): The line object for the validation loss.
    """

    figure: Figure
    axes: Axes
    loss_line: Line2D
    val_loss_line: Line2D

    def __init__(self) -> None:
        super().__init__()

        self.figure = plt.figure(figsize=(10, 5))
        self.axes = self.figure.add_subplot(111)

        self.loss_line = self.axes.plot([], [], label="Loss")[0]
        self.val_loss_line = self.axes.plot([], [], label="Val Loss")[0]
        self.axes.set_title("Loss over epochs")
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Loss")
        self.axes.legend()

    async def on_epoch_end_async(  # pylint: disable=missing-function-docstring
        self,
        neural_network: "NeuralNetwork",
        epoch: int,  # pylint: disable=unused-argument
    ) -> None:
        history = neural_network.history
        loss = [h["loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        self.loss_line.set_data(range(len(loss)), loss)
        self.val_loss_line.set_data(range(len(val_loss)), val_loss)
        self.axes.relim()
        self.axes.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    def __str__(self) -> str:
        return "DrawRealTimeLoss"

    def __repr__(self) -> str:
        return self.__str__()
