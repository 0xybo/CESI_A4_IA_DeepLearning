"""
Module for performing grid search on neural network hyperparameters.
"""

from typing import List, Dict, Any, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .callback.early_stopping import EarlyStopping
from .layer import Layer
from .activation.base import ActivationFunction
from .loss.base import LossFunction
from .neural_network import NeuralNetwork
from .evaluation import Evaluation

# from .callback.progress_bar import ProgressBar
# from .callback.draw_real_time_loss import DrawRealTimeLoss


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

    architecture: List[List[LayerParams]]


class Result(TypedDict):
    """
    Result of a grid search combination
    """

    combination: Dict[str, Any]
    metrics: Dict[str, float]
    network: NeuralNetwork


class GridSearch:
    """
    Grid Search class for hyperparameter tuning with optional multi-threading support.
    """

    def __init__(self, num_threads: int = 1):
        """
        Initialize GridSearch with optional multi-threading.

        Args:
            num_threads (int): Number of threads to use for parallel training.
                Defaults to 1 (single-threaded). Use values > 1 for concurrent training.
        """
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        self.num_threads = num_threads

    def __generate_combinations(self, params: Params) -> List[Dict[str, Any]]:
        """
        Generates all combinations of the parameters for the grid search.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a unique
                combination of parameters.
        """

        keys = [key for key in params.keys() if key not in ["name"]]
        values = (params[key] for key in keys)
        combinations = [
            dict(zip(keys, combination)) for combination in product(*values)
        ]
        return combinations

    def _train_combination(
        self,
        combination_index: Tuple[
            int, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> Result:
        """
        Train a single combination of hyperparameters (worker method for threading).

        Args:
            combination_index: Tuple of (index, combination, x_train, y_train, x_val, y_val)

        Returns:
            Result: A Result dictionary containing the combination, metrics, and trained network.
        """
        _, combination, x_train, y_train, x_val, y_val = combination_index

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

        network = NeuralNetwork(
            layers=layers, loss=combination["loss"], inputs=x_train.shape[1]
        )

        # Add Early Stopping
        network.add_callback(
            EarlyStopping(patience=combination["early_stopping_patience"])
        )

        # Train the network
        network.fit(
            x_train,
            y_train,
            learning_rate=combination["learning_rate"],
            batch_size=combination["batch_size"],
            epochs=combination["epochs"],
            validation_split=0.2,
        )

        # Validation data evaluation
        evaluation = Evaluation(x_val, y_val)
        metrics = evaluation.validate(network)

        # Store the results
        return {
            "combination": combination,
            "metrics": metrics,
            "network": network,
        }

    def search(
        self,
        params: Params,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> List[Result]:
        """
        Performs grid search on the given parameters and evaluates the performance of each
        combination on the validation set.

        Args:
            params (Params): The parameters for the grid search.
            x_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels.
            x_val (np.ndarray): The validation data features.
            y_val (np.ndarray): The validation data labels.
        Returns:
            List[Result]: A list of results for each combination of parameters.
        """
        combinations = self.__generate_combinations(params)

        # Prepare tasks: (index, combination, x_train, y_train, x_val, y_val)
        tasks = [
            (i, combination, x_train, y_train, x_val, y_val)
            for i, combination in enumerate(combinations)
        ]

        result: List[Result] = [None] * len(combinations)

        if self.num_threads == 1:
            # Single-threaded execution with progress printing
            for i, task in enumerate(tasks):
                print(f"Training combination {i + 1}/{len(combinations)}", end="\r")
                result[task[0]] = self._train_combination(task)
                print(f"Completed combination {i + 1}/{len(combinations)}")
        else:
            # Multi-threaded execution
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._train_combination, task): task[0]
                    for task in tasks
                }

                completed = 0
                for future in futures:
                    future_index = futures[future]
                    result[future_index] = future.result()
                    completed += 1
                    print(f"Completed combination {completed}/{len(combinations)}")

        return result

    def __get_combination_name(self, combination: Dict[str, Any]) -> str:
        """
        Generates a name for a combination of parameters for easier identification in results.

        Args:
            combination (Dict[str, Any]): A dictionary containing a combination of parameters.

        Returns:
            str: A string representing the name of the combination.
        """
        name_parts = []
        for key, value in combination.items():
            if key == "architecture":
                arch_str = (
                    "Arch("
                    + "->".join(
                        [
                            f"{layer['neurons'][0]}N-{layer['activation'][0].__class__.__name__}"
                            for layer in value
                        ]
                    )
                    + ")"
                )
                name_parts.append(arch_str)
            else:
                name_parts.append(f"{key}={value}")
        return ", ".join(name_parts)

    def search_and_compare(
        self,
        params: Params,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> pd.DataFrame:
        """
        Performs grid search and compares the results based on validation metrics.

        This method draws a comparison of the different combinations of parameters based on their
        performance on the validation set. It can be used to visualize the results of the grid
        search and identify the best combination of parameters.

        Charts:
            - Bar chart comparing the validation accuracy of each combination of parameters.
            - Bar chart comparing the precision of each combination of parameters.
            - Bar chart comparing the recall of each combination of parameters.
            - Bar chart comparing the F1-score of each combination of parameters.
            - Bar chart comparing the validation loss of each combination of parameters.
            - ROC curves.

        Args:
            params (Params): The parameters for the grid search.
            x_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels.
            x_val (np.ndarray): The validation data features.
            y_val (np.ndarray): The validation data labels.
        """

        results = self.search(params, x_train, y_train, x_val, y_val)

        fig = plt.figure(figsize=(12, 8))

        full_names = tuple(
            self.__get_combination_name(result["combination"]) for result in results
        )
        names = tuple(str(i) for i in range(1, len(full_names) + 1))

        # Accuracy
        ax_accuracy = fig.add_subplot(3, 2, 1)
        ax_accuracy.bar(
            names,
            [result["metrics"]["Accuracy"] for result in results],
        )
        ax_accuracy.set_title("Validation Accuracy")
        ax_accuracy.set_xticklabels(names)

        # Precision
        ax_precision = fig.add_subplot(3, 2, 2)
        ax_precision.bar(
            names,
            [result["metrics"]["Precision"] for result in results],
        )
        ax_precision.set_title("Precision")
        ax_precision.set_xticklabels(names)

        # Recall
        ax_recall = fig.add_subplot(3, 2, 3)
        ax_recall.bar(
            names,
            [result["metrics"]["Recall"] for result in results],
        )
        ax_recall.set_title("Recall")
        ax_recall.set_xticklabels(names)

        # F1 Score
        ax_f1 = fig.add_subplot(3, 2, 4)
        ax_f1.bar(
            names,
            [result["metrics"]["F1 Score"] for result in results],
        )
        ax_f1.set_title("F1 Score")
        ax_f1.set_xticklabels(names)

        # Validation Loss
        ax_loss = fig.add_subplot(3, 2, 5)
        ax_loss.bar(
            names,
            [result["metrics"]["Loss"] for result in results],
        )
        ax_loss.set_title("Validation Loss")
        ax_loss.set_xticklabels(names)

        # ROC Curves
        ax_roc = fig.add_subplot(3, 2, 6)
        evaluation = Evaluation(x_val, y_val)
        for i, result in enumerate(results):
            fpr, tpr = evaluation.calculate_roc_points(result["network"])
            ax_roc.plot(fpr, tpr, label=f"{i+1} (AUC: {result['metrics']['AUC']:.2f})")
        ax_roc.plot(
            [0, 1], [0, 1], "k--", label="Random Classifier"
        )  # Diagonal line for reference
        ax_roc.set_title("ROC Curves")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        # ax_roc.legend()

        # List all combinations with their corresponding names
        output_dict = {
            "Combination": names,
            **{
                metric: [result["metrics"][metric] for result in results]
                for metric in results[0]["metrics"].keys()
            },
            **{
                param: [str(result["combination"][param]) for result in results]
                for param in results[0]["combination"].keys()
            },
        }

        output = pd.DataFrame(output_dict)

        print(output)

        plt.tight_layout()
        plt.show()

        return results
