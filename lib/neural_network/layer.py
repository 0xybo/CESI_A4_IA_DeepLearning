"""
azeazezae
"""

from __future__ import annotations
import numpy as np
from .activation.base import ActivationFunction

# class Layer {
#     + int neurons
#     + ndarray weights
#     + float dropout_rate
#     + ActivationFunction activation

#     + None __init__(int neurons, float dropout_rate, ActivationFunction activation)
#     + ndarray forward(ndarray inputs)
#     + ndarray backward(ndarray dz)
# }

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

    def __init__(
            self,
            neurons: int,
            dropout_rate: float,
            activation: ActivationFunction,
            nb_inputs: int = None
        ) -> None:
        """
        Initializes the Layer instance.

        Args:
            neurons (int): The number of neurons in the layer.
            dropout_rate (float): The dropout rate for regularization.
            activation (ActivationFunction): The activation function used in the layer.
            nb_inputs (int): The number of input features, None for not specified (default: None).
        """
        self.neurons = neurons
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.weights = np.random.rand(neurons, nb_inputs) if nb_inputs is not None else np.array([])
        self.bias = np.random.rand(neurons, 1)

    def set_nb_inputs(self, nb_inputs: int) -> None:
        """
        Sets the number of input features and initializes the weights accordingly.

        Args:
            nb_inputs (int): The number of input features.
        """
        self.weights = np.random.rand(self.neurons, nb_inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer. Input must be of shape (nb_inputs, 1).

        Args:
            inputs (np.ndarray): The input data to the layer.
        """
        self.last_inputs = inputs
        z = self.weights @ inputs + self.bias
        self.last_aggregation_values = z
        return self.activation.compute(z)

    def backward(self, product_last: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs the backward pass through the layer.
        product_last must be of shape (neurons, 1).

        Args:
            product_last (np.ndarray): The product of the gradient of the loss with respect 
                to the output of the previous layer and the weights of the previous layer.
            learning_rate (float): The learning rate for updating the weights.

        Returns:
            np.ndarray: The product of the gradient of the loss with respect to the output of the
              layer and the weights of the layer.
        """
        # dz is neurones lines and 1 column.
        dz = product_last * self.activation.derivative(self.last_aggregation_values)

        # dw is neurons lines and nb_inputs columns.
        dw = dz @ self.last_inputs.T
        db = dz

        self.weights -= dw * learning_rate
        self.bias -= db * learning_rate

        # return a matrix of shape (nb_inputs, 1)
        return self.weights.T @ dz

    def first_layer_backward(self, cost_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs the backward pass for the first layer of the network.
        cost_gradient must be of shape (output neurons, 1).

        Args:
            cost_gradient (np.ndarray): The gradient of the cost function 
                with respect to the output of the first layer.
            learning_rate (float): The learning rate for updating the weights.
        
        Returns:
            np.ndarray: The product of the gradient of the loss with respect to the output of the
              layer and the weights of the layer.
        """
        # dz is neurones lines and 1 column.
        dz = cost_gradient * self.activation.derivative(self.last_aggregation_values)

        # dw is neurons lines and nb_inputs columns.
        dw = dz @ self.last_inputs.T
        db = dz

        self.weights -= dw * learning_rate
        self.bias -= db * learning_rate

        # return a matrix of shape (nb_inputs, 1)
        return self.weights.T @ dz
