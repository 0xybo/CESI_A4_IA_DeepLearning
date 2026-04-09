from typing import List, Tuple, Dict, Any, TypedDict

import pandas as pd
import numpy as np

from lib.neural_network.callback.early_stopping import EarlyStopping
from lib.neural_network.layer import Layer
from .activation.base import ActivationFunction
from .activation.sigmoid import Sigmoid
from .loss.base import LossFunction
from .neural_network import NeuralNetwork
from .evaluation import Evaluation


class LayerParams(TypedDict):
    """
    Parameters for a layer in the neural network
    """

    neurons: List[int]
    dropout_rate: List[float]
    activation: List[ActivationFunction]


class Params(TypedDict):
    """
    Parameters for the grid search
    """

    learning_rate: List[float]
    batch_size: List[int]
    epochs: List[int]

    loss: List[LossFunction]

    early_stopping_patience: List[int]
    early_stopping_delta: List[float]

    architecture: List[List[LayerParams]]


class Result(TypedDict):
    """
    Result of a grid search combination
    """

    combination: Dict[str, Any]
    metrics: pd.DataFrame


class GridSearch:
    """
    Grid Search class for hyperparameter tuning
    """

    def __generate_combinations(self, params: Params) -> List[Dict[str, Any]]:
        """
        Generates all combinations of the parameters for the grid search.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a unique combination of parameters.
        """
        from itertools import product

        keys = params.keys()
        values = (params[key] for key in keys)
        combinations = [
            dict(zip(keys, combination)) for combination in product(*values)
        ]
        return combinations

    def search(
        self,
        params: Params,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> List[Result]:
        combinations = self.__generate_combinations(params)

        # Évaluation sur les données de validation
        evaluation = Evaluation(X_val, y_val)
        result: List[Result] = []

        for combination in combinations:
            layers = []

            for layer_params in combination["architecture"]:
                for neurons, dropout_rate, activation in zip(
                    layer_params["neurons"],
                    layer_params["dropout_rate"],
                    layer_params["activation"],
                ):
                    layers.append(
                        Layer(
                            neurons=neurons,
                            dropout_rate=dropout_rate,
                            activation=activation,
                        )
                    )

            network = NeuralNetwork(layers=layers, loss=combination["loss"], inputs=X_train.shape[1])

            # Ajouter Early Stopping
            network.add_callback(
                EarlyStopping(patience=combination["early_stopping_patience"])
            )

            # Entraîner le réseau
            network.fit(
                X_train,
                y_train,
                learning_rate=combination["learning_rate"],
                batch_size=combination["batch_size"],
                epochs=combination["epochs"],
                validation_split=0.2,
            )

            # Évaluation sur les données de validation
            metrics = evaluation.validate(network)

            # Stocker les résultats
            result.append({"combination": combination, "metrics": metrics})

        return result
