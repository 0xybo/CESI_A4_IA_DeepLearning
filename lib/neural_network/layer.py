"""
Layer module
"""

from __future__ import annotations
import numpy as np
from .activation.base import ActivationFunction

class Layer:
    """
    Class representing a layer in a neural network.
    The weights matrix is of shape (neurons, nb_inputs)
    The bias vector is of shape (neurons, 1)

    Attributes:
        neurons (int): The number of neurons in the layer.
        weights (np.ndarray): The weights of the layer.
        bias (np.ndarray): The bias values of the layer.
        dropout_rate (float): The dropout rate for regularization.
        activation (ActivationFunction): The activation function used in the layer.
    """

    neurons: int
    weights: np.ndarray
    bias: np.ndarray
    dropout_rate: float
    activation: ActivationFunction
    last_aggregation_values: np.ndarray
    last_inputs: np.ndarray
    dropout_mask: np.ndarray

    def __init__(
        self,
        neurons: int,
        activation: ActivationFunction,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Initializes the Layer instance.

        Args:
            neurons (int): The number of neurons in the layer.
            dropout_rate (float): The dropout rate for regularization.
            activation (ActivationFunction): The activation function used in the layer.
        """
        self.neurons = neurons
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.weights = np.array([])
        self.bias = np.zeros((neurons, 1))

    def set_nb_inputs(self, nb_inputs: int) -> None:
        """
        Sets the number of input features and initializes the weights accordingly.
        Uses He initialization for better convergence with ReLU activations.

        Args:
            nb_inputs (int): The number of input features.
        """
        self.weights = np.random.randn(self.neurons, nb_inputs) * np.sqrt(
            2.0 / nb_inputs
        )

    def forward(self, inputs: np.ndarray, training: bool) -> np.ndarray:
        """
        Performs the forward pass through the layer. Input must be of shape (nb_inputs, batches).

        Args:
            inputs (np.ndarray): The input data to the layer of shape (nb_inputs, batch_size).
        """
        self.last_inputs = inputs
        z = self.weights @ inputs + self.bias
        self.last_aggregation_values = z
        a = self.activation.compute(z)

        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
            a *= self.dropout_mask
            a /= (1.0 - self.dropout_rate)

        return a

    def backward(self, product_last: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs the backward pass through the layer.
        product_last must be of shape (neurons, batches).

        Args:
            product_last (np.ndarray): Gradient from next layer, shape (neurons, batch_size).
            learning_rate (float): The learning rate for updating the weights.

        Returns:
            np.ndarray: Gradient to pass to previous layer, shape (nb_inputs, batch_size).
        """
        # dz is neurones lines and batches columns.
        dz = product_last * self.activation.derivative(self.last_aggregation_values)

        if self.dropout_rate > 0:
            dz *= self.dropout_mask
            dz /= (1.0 - self.dropout_rate)

        # dw is neurons lines and nb_inputs columns.
        dw = dz @ self.last_inputs.T
        db = np.sum(dz, axis=1, keepdims=True)

        # Gradient for previous layer: shape (nb_inputs, batch_size)
        grad = self.weights.T @ dz
        
        # Update weights and biases
        self.weights -= dw * learning_rate
        self.bias -= db * learning_rate

        return grad
