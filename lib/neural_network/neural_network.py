"""
Neural Network class
"""

from __future__ import annotations
from typing import List, Optional, TypedDict
import numpy as np
from .layer import Layer
from .callback.base import Callback
from .loss import LossFunction


class History(TypedDict):
    """
    History of the training process
    """

    loss: float
    val_loss: float
    y_pred: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray


class NeuralNetwork:
    """
    Neural Network class

    Attributes:
    layers: List[Layer] - list of layers in the neural network
    loss: Optional[LossFunction] - loss function used to train the neural network
    callbacks: List[Callback] - list of callbacks used during training
    fiting: bool - whether the neural network is currently being trained
    history: Optional[History] - history of the training process
    x_train: Optional[np.ndarray] - training data
    y_train: Optional[np.ndarray] - training labels
    epochs: Optional[int] - number of epochs to train the neural network
    epoch: Optional[int] - current epoch during training
    batch_size: Optional[int] - batch size used during training
    validation_split: Optional[float] - percentage of training data used for validation
    learning_rate: Optional[float] - learning rate used during training
    """

    layers: List[Layer]
    loss: LossFunction
    callbacks: List[Callback]
    fiting: bool
    history: List[History]
    x_train: Optional[np.ndarray]
    y_train: Optional[np.ndarray]
    epochs: Optional[int]
    epoch: Optional[int]
    batch_size: Optional[int]
    validation_split: Optional[float]
    learning_rate: Optional[float]
    inputs: int

    def __init__(self, layers: List[Layer], loss: LossFunction, inputs: int) -> None:
        self.layers = layers
        self.loss = loss
        self.callbacks = []
        self.fiting = False
        self.history = []
        self.x_train = None
        self.y_train = None
        self.epochs = None
        self.epoch = None
        self.batch_size = None
        self.validation_split = None
        self.learning_rate = None
        self.inputs = inputs

        self.__param_layers()

    def add_layer(self, layer: Layer) -> None:
        """
        Add a layer to the neural network

        Args:
        layer: Layer - layer to add to the neural network
        """

        self.layers.append(layer)
        self.__param_layers()

    def add_callback(self, callback: Callback) -> None:
        """
        Add a callback to the neural network

        Args:
        callback: Callback - callback to add to the neural network
        """

        self.callbacks.append(callback)

    def __param_layers(self) -> None:
        self.layers[0].set_nb_inputs(self.inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].set_nb_inputs(self.layers[i - 1].neurons)

    def __callbacks(self, event: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            getattr(callback, event)(self, *args, **kwargs)
            getattr(callback, f"{event}_async")(self, *args, **kwargs)

    def predict(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Make a prediction with the neural network

        Args:
        x: np.ndarray - input data

        Returns:
        np.ndarray - predicted output
        """

        x = x.T
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def predicts(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Make predictions with the neural network

        Args:
        x: np.ndarray - input data

        Returns:
        np.ndarray - predicted output
        """

        predictions = []
        for i in range(x.shape[0]):
            predictions.append(self.predict(x[i : i + 1], training=training))
        return np.array(predictions).squeeze()

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split: float,
        learning_rate: float,
    ) -> None:
        """
        Train the neural network

        Features :
        - Cross-validation : split the training data into training and validation sets and evaluate
          the model on the validation set at each epoch

        Args:
            x_train: np.ndarray - training data
            y_train: np.ndarray - training labels
            epochs: int - number of epochs to train the neural network
            batch_size: int - batch size used during training
            validation_split: float - percentage of training data used for validation
            learning_rate: float - learning rate used during training
        """
        self.fiting = True
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate

        self.history = []

        self.__callbacks("on_train_begin")

        for epoch in range(epochs):
            # Check if the training has been cancelled by a callback
            # For example, a callback like EarlyStopping can set
            # self.fiting to False to stop the training process early
            if not self.fiting:
                self.__callbacks("on_train_cancel")
                break

            self.epoch = epoch

            self.__callbacks("on_epoch_begin", epoch)

            # Shuffle the training data
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            # Split the training data into training and validation sets
            split_index = int(x_train.shape[0] * (1 - validation_split))
            x_train_split = x_train[:split_index]
            y_train_split = y_train[:split_index]
            x_val_split = x_train[split_index:]
            y_val_split = y_train[split_index:]

            loss_value = 0.0
            x_batch: np.ndarray = np.array([])
            y_batch: np.ndarray = np.array([])

            # Train the model on the training set
            for i in range(0, x_train_split.shape[0], batch_size):
                self.__callbacks("on_batch_begin", i // batch_size)

                x_batch = x_train_split[i : i + batch_size]
                y_batch = y_train_split[i : i + batch_size]

                # Forward pass
                y_pred = self.predicts(x_batch, training=True)

                # Compute loss and gradients
                loss_value = self.loss.compute(y_batch, y_pred)
                loss_gradients = self.loss.derivative(y_batch, y_pred)

                # Backward pass
                for layer in reversed(self.layers):
                    loss_gradients = layer.backward(loss_gradients, learning_rate)

                self.__callbacks("on_batch_end", i // batch_size)

            # Evaluate the model on the validation set
            val_pred = self.predicts(x_val_split, training=False)
            val_loss_value = self.loss.compute(y_val_split, val_pred)

            # Save history
            self.history.append(
                {
                    "loss": loss_value,
                    "val_loss": val_loss_value,
                    "y_pred": val_pred,
                    "x_train": x_batch,
                    "y_train": y_batch,
                }
            )

            self.__callbacks("on_epoch_end", epoch)

        self.__callbacks("on_train_end")
        self.fiting = False
