"""
Neural Network class
"""

from __future__ import annotations
from typing import List, TypedDict
import asyncio
import numpy as np
import threading
from .layer import Layer
from .callback.base import Callback
from .loss import LossFunction
from ..utils.run_coroutine_sync import run_coroutine_sync


class History(TypedDict):
    """
    History of the training process
    """

    loss: float
    val_loss: float
    y_pred: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    learning_rate: float


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
    threshold: float - threshold used for binary classification
    """

    layers: List[Layer]
    loss: LossFunction
    callbacks: List[Callback]
    fiting: bool
    history: List[History]
    x_train: np.ndarray
    y_train: np.ndarray
    epochs: int
    epoch: int
    batch_size: int
    validation_split: float
    learning_rate: float
    threshold: float
    inputs: int
    trained: bool = False
    rng: np.random.Generator

    def __init__(
        self,
        layers: List[Layer],
        loss: LossFunction,
        inputs: int,
        seed: int | None = None,
        name: str = "NeuralNetwork",
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.callbacks = []
        self.fiting = False
        self.history = []
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.epochs = 0
        self.epoch = 0
        self.batch_size = 0
        self.validation_split = 0.0
        self.learning_rate = 0.0
        self.threshold = 0.5
        self.inputs = inputs
        self.name = name
        # Use thread-specific RNG to avoid contention in multi-threaded scenarios
        self.rng = np.random.default_rng(seed)
        self.__param_layers()

    def to_dict(self) -> dict:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "loss": str(self.loss),
            "callbacks": [str(callback) for callback in self.callbacks],
            "fiting": self.fiting,
            # "history": self.history,
            # "x_train": self.x_train.tolist(),
            # "y_train": self.y_train.tolist(),
            "epochs": self.epochs,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "learning_rate": self.learning_rate,
            "threshold": self.threshold,
            "inputs": self.inputs,
            "trained": self.trained,
        }

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
            # Call the asynchronous version of the callback if it exists
            # This asynchronous is not awaited, it is just called in parallel
            # This allows to not block the training process while the callback is running
            # asyncio.create_task(
            #     getattr(callback, f"{event}_async")(self, *args, **kwargs)
            # )

    def predict(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Make a prediction with the neural network

        Args:
        x: np.ndarray - input data

        Returns:
        np.ndarray - predicted output
        """

        if not self.trained:
            raise ValueError(
                "Neural network not trained yet. Please train the model before making predictions."
            )

        return self.__predict(x, training=training)

    def predict_proba(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Make a probability prediction with the neural network

        Args:
        x: np.ndarray - input data

        Returns:
        np.ndarray - predicted probabilities
        """

        if not self.trained:
            raise ValueError(
                "Neural network not trained yet. Please train the model before making predictions."
            )

        return self.__predict_proba(x, training=training)

    def __predict(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Make a prediction with the neural network

        Args:
        x: np.ndarray - input data

        Returns:
        np.ndarray - predicted output
        """

        return (self.__predict_proba(x, training=training) > self.threshold).astype(int)

    def __predict_proba(self, x: np.ndarray, training: bool = False) -> np.ndarray:
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

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        learning_rate: float = 0.01,
        threshold: float = 0.5,
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
            threshold: float - threshold used for binary classification
        """

        # Check if the training is already in progress
        if self.fiting:
            raise ValueError(
                "Training is already in progress. Please wait for it to finish."
            )

        self.__fit(
            x_train,
            y_train,
            epochs,
            batch_size,
            validation_split,
            learning_rate,
            threshold,
        )

    def __fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        learning_rate: float = 0.01,
        threshold: float = 0.5,
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
            threshold: float - threshold used for binary classification
        """
        self.fiting = True
        self.trained = False
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.history = []

        self.__callbacks("on_train_begin")

        # Shuffle the training data
        indices = np.arange(self.x_train.shape[0])
        self.rng.shuffle(indices)
        x_train = self.x_train[indices]
        y_train = self.y_train[indices]

        for epoch in range(self.epochs):
            # Check if the training has been cancelled by a callback
            # For example, a callback like EarlyStopping can set
            # self.fiting to False to stop the training process early
            if not self.fiting:
                self.__callbacks("on_train_cancel")
                break

            self.epoch = epoch

            self.__callbacks("on_epoch_begin", epoch)

            # Split the training data into training and validation sets
            split_index = int(self.x_train.shape[0] * (1 - self.validation_split))
            x_train_split = self.x_train[:split_index]
            y_train_split = self.y_train[:split_index]
            x_val_split = self.x_train[split_index:]
            y_val_split = self.y_train[split_index:]

            loss_value = 0.0
            x_batch: np.ndarray = np.array([])
            y_batch: np.ndarray = np.array([])

            # Train the model on the training set
            for i in range(0, x_train_split.shape[0], self.batch_size):
                self.__callbacks("on_batch_begin", i // self.batch_size)

                x_batch = x_train_split[i : i + self.batch_size]
                y_batch = y_train_split[i : i + self.batch_size]

                # Forward pass
                y_pred = self.__predict_proba(x_batch, training=True)

                # Compute loss and gradients
                loss_value = self.loss.compute(y_batch.reshape(1, -1), y_pred)
                loss_gradients = self.loss.derivative(y_batch.reshape(1, -1), y_pred)

                # Backward pass
                for layer in reversed(self.layers):
                    loss_gradients = layer.backward(loss_gradients, self.learning_rate)

                self.__callbacks("on_batch_end", i // self.batch_size)

            # Evaluate the model on the validation set
            val_pred = self.__predict_proba(x_val_split, training=False)
            val_loss_value = self.loss.compute(y_val_split.reshape(1, -1), val_pred)

            # Save history
            self.history.append(
                {
                    "loss": loss_value,
                    "val_loss": val_loss_value,
                    "x_train": x_batch,
                    "y_train": y_batch,
                    "y_pred": val_pred,
                    "learning_rate": self.learning_rate,
                }
            )

            self.__callbacks("on_epoch_end", epoch)

        self.__callbacks("on_train_end")
        self.fiting = False
        self.trained = True

    def __str__(self) -> str:
        """
        String representation of the neural network
        """
        result = "NeuralNetwork("
        for i, layer in enumerate(self.layers):
            result += f"{layer.neurons}:{layer.activation}"
            if i < len(self.layers) - 1:
                result += "->"
        result += ")"
        return result

    def __repr__(self) -> str:
        return self.__str__()
