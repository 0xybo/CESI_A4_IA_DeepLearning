"""
Module for performing grid search on neural network hyperparameters.
"""

import threading
from typing import List, Dict, Any, TypedDict, Tuple, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

from .layer import Layer
from .evaluation import Evaluation
from .loss.base import LossFunction
from .neural_network import NeuralNetwork
from .activation.base import ActivationFunction
from .callback.early_stopping import EarlyStopping
from .callback.train_progress_bar import TrainProgressBar
from .callback.adaptive_learning_rate_reduce_on_plateau import ReduceOnPlateau
from .callback.adaptive_learning_rate_step_decay import StepDecay
from .callback.base import Callback

# Lock for thread-safe matplotlib operations
_matplotlib_lock = threading.Lock()


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
    learning_rate_adaptive: List[Callback or None]

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

    num_threads: int = 1

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

    def __train_combination(
        self,
        combination_index: Tuple[
            int, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int
        ],
    ) -> Result:
        """
        Train a single combination of hyperparameters (worker method for threading).

        Args:
            combination_index: Tuple of (index, combination, x_train, y_train, x_val, y_val, seed)

        Returns:
            Result: A Result dictionary containing the combination, metrics, and trained network.
        """
        _, combination, x_train, y_train, x_val, y_val, seed = combination_index

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
                        seed=seed,  # Pass unique seed to layer
                    )
                )

        network = NeuralNetwork(
            layers=layers,
            loss=combination["loss"],
            inputs=x_train.shape[1],
            seed=seed,  # Pass unique seed to network
            name=f"Combination {combination_index[0]+1}",
        )

        network.add_callback(
            EarlyStopping(patience=combination["early_stopping_patience"])
        )

        if combination["learning_rate_adaptive"] is not None:
            network.add_callback(combination["learning_rate_adaptive"])

        network.add_callback(
            TrainProgressBar(
                position=combination_index[0] + 1
            )  # Position progress bar based on index
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

    def generate_combinations(self, params: Params) -> List[Dict[str, Any]]:
        """
        Generates all combinations of the parameters for the grid search.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a unique
                combination of parameters.
        """

        keys = list(params.keys())
        values = (params[key] for key in keys)

        return [dict(zip(keys, combination)) for combination in product(*values)]

    def get_number_of_combinations(self, params: Params) -> int:
        """
        Calculates the total number of combinations of parameters for the grid search.

        Args:
            params (Params): The parameters for the grid search.
        Returns:
            int: The total number of combinations.
        """

        total_combinations = 1
        for key in params.keys():
            total_combinations *= len(params[key])

        return total_combinations

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
        combinations = self.generate_combinations(params)

        # Generate unique seeds for each combination to avoid RNG contention in threads
        rng = np.random.default_rng(42)
        seeds = rng.integers(0, 2**31, size=len(combinations))

        # Prepare tasks: (index, combination, x_train, y_train, x_val, y_val, seed)
        tasks = [
            (i, combination, x_train, y_train, x_val, y_val, seeds[i])
            for i, combination in enumerate(combinations)
        ]

        result: List[Result] = [None] * len(combinations)

        if self.num_threads == 1:
            # Single-threaded execution with tqdm progress bar
            for task in tqdm(tasks, desc="Grid Search", unit="combination", position=0):
                result[task[0]] = self.__train_combination(task)
        else:
            # Multi-threaded execution with tqdm progress bar
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self.__train_combination, task): task[0]
                    for task in tasks
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Grid Search",
                    unit="combination",
                    position=0,
                ):
                    future_index = futures[future]
                    result[future_index] = future.result()

        return result

    def search_and_compare(
        self,
        params: Params,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        date: str = "",
        draw: bool = True,
    ) -> List[Result]:
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

        # Use a lock to ensure thread-safe matplotlib operations
        # This prevents race conditions and global state corruption from multiple threads
        with _matplotlib_lock:
            # +-----------------+-----------------+-----------------+
            # |    Accuracy     |    Precision    |      Recall     |
            # +-----------------+-----------------+-----------------+
            # |     F1 Score    |     F2 Score    | Validation Loss |
            # +-----------------+-----------------+-----------------+
            # |           Learning rate           |      Legend     |
            # +-----------------------------------+-----------------+
            # |                Training Loss Curves                 |
            # +-----------------------------------+-----------------+
            # |                 Testing Loss Curves                 |
            # +-----------------------------------------------------+
            # |                     ROC Curves                      |
            # +-----------------------------------------------------+

            fig, axes = plt.subplots(6, 3, figsize=(8, 15), dpi=80)

            full_names = tuple(
                self.__get_combination_name(result["combination"]) for result in results
            )
            names = tuple(str(i) for i in range(1, len(full_names) + 1))

            # Print combination names and metrics for easier identification
            text = []
            for i, (full_name, result) in enumerate(zip(full_names, results)):
                text.append(f"Combination {i+1}: {full_name}")
            max_length = max(len(line) for line in text)
            if draw:
                print("")
                print("=" * max_length)
                for line in text:
                    print(f"{line:<{max_length}}")
                print("=" * max_length)
            else:
                with open(
                    f"./grid_search_results/{date}.txt", "w", encoding="utf-8"
                ) as f:
                    f.write("\n".join(text))

            # Accuracy
            ax_accuracy = axes[0, 0]
            ax_accuracy.bar(
                names,
                [result["metrics"]["Accuracy"] for result in results],
            )
            ax_accuracy.set_title("Validation Accuracy")
            for i, full_name in enumerate(full_names):
                ax_accuracy.text(
                    i,
                    max(result["metrics"]["Accuracy"] for result in results) * 1.01,
                    f"{results[i]["metrics"]["Accuracy"]:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            ax_accuracy.set_ylim(
                min(result["metrics"]["Accuracy"] for result in results) / 1.01,
                max(result["metrics"]["Accuracy"] for result in results) * 1.2,
            )
            # ax_accuracy.set_xticklabels(names)

            # Precision
            ax_precision = axes[0, 1]
            ax_precision.bar(
                names,
                [result["metrics"]["Precision"] for result in results],
            )
            ax_precision.set_title("Precision")
            for i, full_name in enumerate(full_names):
                ax_precision.text(
                    i,
                    max(result["metrics"]["Precision"] for result in results) * 1.01,
                    f"{results[i]["metrics"]["Precision"]:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            ax_precision.set_ylim(
                min(result["metrics"]["Precision"] for result in results) / 1.01,
                max(result["metrics"]["Precision"] for result in results) * 1.2,
            )
            # ax_precision.set_xticklabels(names)

            # Recall
            ax_recall = axes[0, 2]
            ax_recall.bar(
                names,
                [result["metrics"]["Recall"] for result in results],
            )
            ax_recall.set_title("Recall")
            for i, full_name in enumerate(full_names):
                ax_recall.text(
                    i,
                    max(result["metrics"]["Recall"] for result in results) * 1.01,
                    f"{results[i]["metrics"]["Recall"]:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            ax_recall.set_ylim(
                min(result["metrics"]["Recall"] for result in results) / 1.01,
                max(result["metrics"]["Recall"] for result in results) * 1.2,
            )
            # ax_recall.set_xticklabels(names)

            # F1 Score
            ax_f1 = axes[1, 0]
            ax_f1.bar(
                names,
                [result["metrics"]["F1 Score"] for result in results],
            )
            ax_f1.set_title("F1 Score")
            for i, full_name in enumerate(full_names):
                ax_f1.text(
                    i,
                    max(result["metrics"]["F1 Score"] for result in results) * 1.01,
                    f"{results[i]["metrics"]["F1 Score"]:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            ax_f1.set_ylim(
                min(result["metrics"]["F1 Score"] for result in results) / 1.01,
                max(result["metrics"]["F1 Score"] for result in results) * 1.2,
            )
            # ax_f1.set_xticklabels(names)

            # F2 Score
            ax_f2 = axes[1, 1]
            ax_f2.bar(
                names,
                [result["metrics"]["F2 Score"] for result in results],
            )
            ax_f2.set_title("F2 Score")
            for i, full_name in enumerate(full_names):
                ax_f2.text(
                    i,
                    max(result["metrics"]["F2 Score"] for result in results) * 1.01,
                    f"{results[i]["metrics"]["F2 Score"]:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            ax_f2.set_ylim(
                min(result["metrics"]["F2 Score"] for result in results) / 1.01,
                max(result["metrics"]["F2 Score"] for result in results) * 1.2,
            )
            # ax_f2.set_xticklabels(names)

            # Validation Loss
            ax_loss = axes[1, 2]
            ax_loss.bar(
                names,
                [result["metrics"]["Loss"] for result in results],
            )
            ax_loss.set_title("Validation Loss")
            for i, full_name in enumerate(full_names):
                ax_loss.text(
                    i,
                    max(result["metrics"]["Loss"] for result in results) * 1.01,
                    f"{results[i]["metrics"]["Loss"]:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            ax_loss.set_ylim(
                min(result["metrics"]["Loss"] for result in results) / 1.01,
                max(result["metrics"]["Loss"] for result in results) * 1.2,
            )
            # ax_loss.set_xticklabels(names)

            # Learning rate curves
            gs_lr_curves = axes[2, 0].get_gridspec()
            for ax in axes[2, :-1]:
                ax.remove()
            ax_lr_curves = fig.add_subplot(gs_lr_curves[2, :-1])
            lines = []
            for i, result in enumerate(results):
                history = result["network"].history
                if history:
                    lines += ax_lr_curves.plot(
                        [entry["learning_rate"] for entry in history],
                        label=f"{i+1} (Final: {history[-1]['learning_rate']:.4e})",
                    )
            ax_lr_curves.set_title("Learning Rate Curves")
            ax_lr_curves.set_xlabel("Epoch")
            ax_lr_curves.set_ylabel("Learning Rate")

            # Legend
            ax_legend = axes[2, 2]
            ax_legend.axis("off")
            ax_legend.legend(
                lines,
                [f"Combination {i+1}" for i in range(len(results))],
                loc="center",
                fontsize=8,
            )

            # Loss Curves
            gs_loss_curves = axes[3, 0].get_gridspec()
            for ax in axes[3, 0:]:
                ax.remove()
            ax_loss_curves = fig.add_subplot(gs_loss_curves[3, :])
            for i, result in enumerate(results):
                history = result["network"].history
                if history:
                    ax_loss_curves.plot(
                        [entry["loss"] for entry in history],
                        label=f"{i+1} (Final: {history[-1]['loss']:.4f})",
                    )
            ax_loss_curves.set_title("Training Loss Curves")
            ax_loss_curves.set_xlabel("Epoch")
            ax_loss_curves.set_ylabel("Loss")

            # Testing Loss Curves
            gs_test_loss_curves = axes[4, 0].get_gridspec()
            for ax in axes[4, :]:
                ax.remove()
            ax_test_loss_curves = fig.add_subplot(gs_test_loss_curves[4, :])
            for i, result in enumerate(results):
                history = result["network"].history
                if history:
                    ax_test_loss_curves.plot(
                        [entry["val_loss"] for entry in history],
                        label=f"{i+1} (Final: {history[-1]['val_loss']:.4f})",
                    )
            ax_test_loss_curves.set_title("Testing Loss Curves")
            ax_test_loss_curves.set_xlabel("Epoch")
            ax_test_loss_curves.set_ylabel("Loss")

            # ROC Curves
            gs_roc = axes[5, 0].get_gridspec()
            for ax in axes[5, :]:
                ax.remove()
            ax_roc = fig.add_subplot(gs_roc[5, :])
            evaluation = Evaluation(x_val, y_val)
            for i, result in enumerate(results):
                fpr, tpr, _ = evaluation.calculate_roc_points(result["network"])
                ax_roc.plot(
                    tpr, fpr, label=f"{i+1} (AUC: {result['metrics']['AUC']:.2f})"
                )
            ax_roc.plot(
                [0, 1], [0, 1], "k--", label="Random Classifier"
            )  # Diagonal line for reference
            ax_roc.set_title("ROC Curves")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")

            fig.tight_layout()
            if draw:
                plt.show()
            else:
                fig.savefig(f"./grid_search_results/{date}.png")

            # Close figure to prevent memory leaks
            plt.close(fig)

        return results
