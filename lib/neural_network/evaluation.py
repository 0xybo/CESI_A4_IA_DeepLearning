"""
Evaluation Class  for Neural Networks
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .neural_network import NeuralNetwork


class Evaluation:
    """
    Evaluates the performance of a neural network on a validation dataset.
    This class provides methods to calculate various performance metrics such as accuracy,
    precision, recall, F1 score, and AUC. It also includes a method to draw the ROC curve.

    Attributes:
        neural_network (NeuralNetwork): The neural network to evaluate.
        x_validation (np.ndarray): The validation input data.
        y_validation (np.ndarray): The validation target data.
        __confusion_matrix (Tuple[int, int, int, int]): A tuple containing the counts of
            true positives, false positives, true negatives, and false negatives.
    """

    neural_network: Optional[NeuralNetwork] = None
    x_validation: np.ndarray
    y_validation: np.ndarray
    confusion_matrix_cache: Tuple[int, int, int, int]  # (TP, FP, TN, FN)
    populated_confusion_matrix: bool = False

    @staticmethod
    def __confusion_matrix_guard(func):
        """
        Decorator to ensure that the confusion matrix is calculated before accessing it.
        If the confusion matrix is not populated, it will be calculated using the validation
        data and the neural network's predictions.

        Args:
            func (Callable): The function to decorate.
        Returns:
            Callable: The decorated function that ensures the confusion matrix is populated.
        """

        def wrapper(self: "Evaluation", *args, **kwargs):
            if not self.populated_confusion_matrix:
                self.confusion_matrix()  # Calculate confusion matrix to populate it
            return func(self, *args, **kwargs)

        return wrapper

    def __init__(
        self,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> None:
        """
        Initializes the Evaluation object with optional validation data.
        Args:
            x_validation (np.ndarray): The validation input data.
            y_validation (np.ndarray): The validation target data.
        """
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.confusion_matrix_cache = (
            0,
            0,
            0,
            0,
        )  # Initialize confusion matrix counts to zero

    def set_neural_network(self, neural_network: NeuralNetwork) -> None:
        """
        Sets the neural network to evaluate.
        Args:
            neural_network (NeuralNetwork): The neural network to evaluate.
        """
        self.neural_network = neural_network

    def validate(self, neural_network: Optional[NeuralNetwork] = None) -> Dict:
        """
        Validates the neural network and returns a dictionary containing the calculated metrics.
        Args:
            neural_network (Optional[NeuralNetwork]): The neural network to validate.
        Returns:
            Dict: A dictionary containing the calculated metrics (accuracy, precision, recall,
                F1 score, and AUC).
        """
        if neural_network is not None:
            self.neural_network = neural_network

        if self.neural_network is None:
            raise ValueError("Neural network not set.")

        metrics = {
            "Accuracy": self.accuracy(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "F1 Score": self.f1_score(),
            "AUC": self.auc(),
            "Loss": self.loss(),
            "True Positives": self.confusion_matrix_cache[0],
            "False Positives": self.confusion_matrix_cache[1],
            "True Negatives": self.confusion_matrix_cache[2],
            "False Negatives": self.confusion_matrix_cache[3],
        }

        return metrics

    def loss(self) -> float:
        """
        Calculates and returns the loss of the neural network on the validation data.
        Returns:
            float: The loss value.
        """
        if self.neural_network is None:
            raise ValueError("Neural network not set.")

        y_pred_prob: np.ndarray = self.neural_network.predict_proba(self.x_validation).T
        return self.neural_network.loss.compute(y_pred_prob, self.y_validation)

    @__confusion_matrix_guard
    def accuracy(self) -> float:
        """
        Calculates and returns the accuracy of the neural network.
        Returns:
            float: The accuracy value.
        """
        true_positives: int = self.confusion_matrix_cache[0]
        false_positives: int = self.confusion_matrix_cache[1]
        true_negatives: int = self.confusion_matrix_cache[2]
        false_negatives: int = self.confusion_matrix_cache[3]

        return (true_positives + true_negatives) / (
            true_positives + false_positives + true_negatives + false_negatives
        )

    @__confusion_matrix_guard
    def precision(self) -> float:
        """
        Calculates and returns the precision of the neural network.
        Returns:
            float: The precision value.
        """
        true_positives: int = self.confusion_matrix_cache[0]
        false_positives: int = self.confusion_matrix_cache[1]

        return (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

    @__confusion_matrix_guard
    def recall(self) -> float:
        """
        Calculates and returns the recall of the neural network.
        Returns:
            float: The recall value.
        """
        true_positives: int = self.confusion_matrix_cache[0]
        false_negatives: int = self.confusion_matrix_cache[3]

        return (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

    @__confusion_matrix_guard
    def f1_score(self) -> float:
        """Calculates and returns the F1 score of the neural network.
        Returns:
            float: The F1 score value.
        """
        precision: float = self.precision()
        recall: float = self.recall()

        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    def auc(self) -> float:
        """
        Calculates and returns the AUC of the neural network.
        Returns:
            float: The AUC value.
        """
        true_positives_rate_list, false_positives_rate_list = (
            self.calculate_roc_points()
        )
        return np.trapezoid(true_positives_rate_list, false_positives_rate_list)

    def calculate_roc_points(
        self, neural_network: Optional[NeuralNetwork] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Calculates the true positive rate and false positive rate lists for the ROC curve.
        Args:
            neural_network (NeuralNetwork): The neural network to evaluate.
        Returns:
            Tuple[List[float], List[float]]: The true positive rate and false positive rate lists.
        """
        if neural_network is not None:
            self.neural_network = neural_network

        if self.neural_network is None:
            raise ValueError("Neural network not set.")

        y_pred_prob: np.ndarray = self.neural_network.predict_proba(
            self.x_validation
        ).reshape(-1)
        thresholds = np.linspace(0, 1, num=100)
        true_positives_rate_list: List[float] = []
        false_positives_rate_list: List[float] = []

        for threshold in thresholds:
            y_pred: np.ndarray = (y_pred_prob >= threshold).astype(int)
            tp, fp, tn, fn = self.__calculate_confusion_matrix(
                y_true=self.y_validation, y_pred=y_pred.reshape(self.y_validation.shape)
            )
            true_positives_rate_list.append(
                1 - tp / (tp + fn) if (tp + fn) > 0 else 0.0
            )
            false_positives_rate_list.append(
                1 - fp / (fp + tn) if (fp + tn) > 0 else 0.0
            )

        return true_positives_rate_list, false_positives_rate_list

    def draw_roc(self, ax: Optional[Axes] = None) -> None:
        """
        Draws the ROC curve for the neural network.
        Args:
            ax (Optional[Axes]): An optional matplotlib Axes object to draw the ROC curve on.
                If None, a new figure and axes will be created.
        Returns:
            Tuple[List[float], List[float]]: The true positive rate and false positive rate lists.
        Actions:
            Plots the ROC curve using matplotlib.
        """
        if self.neural_network is None:
            raise ValueError("Neural network not set.")

        true_positives_rate_list, false_positives_rate_list = (
            self.calculate_roc_points()
        )

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        ax.plot(false_positives_rate_list, true_positives_rate_list, label="ROC Curve")
        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        if ax is None:
            plt.show()

    def __calculate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Calculates the confusion matrix based on the true labels and predicted labels.
        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        Returns:
            Tuple[int, int, int, int]: A tuple containing the counts of true positives,
                false positives, true negatives, and false negatives.
        """
        true_positives: int = np.sum((y_pred == 1) & (y_true == 1))
        false_positives: int = np.sum((y_pred == 1) & (y_true == 0))
        true_negatives: int = np.sum((y_pred == 0) & (y_true == 0))
        false_negatives: int = np.sum((y_pred == 0) & (y_true == 1))

        return true_positives, false_positives, true_negatives, false_negatives

    def confusion_matrix(
        self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculates and returns the confusion matrix for the neural network.
        If x and y are not provided, it uses the validation data stored in the object.
        Args:
            x (Optional[np.ndarray]): The input data to use for calculating the confusion matrix.
                Default is None.
            y (Optional[np.ndarray]): The target data to use for calculating the confusion matrix.
                Default is None.
        """
        if x is None:
            x = self.x_validation
        if y is None:
            y = self.y_validation

        if x is None or y is None:
            raise ValueError("Validation data not provided.")

        if self.neural_network is None:
            raise ValueError("Neural network not set.")

        y_pred: np.ndarray = self.neural_network.predict(x).T

        self.confusion_matrix_cache = self.__calculate_confusion_matrix(
            y_true=y, y_pred=y_pred.reshape(y.shape)
        )

        self.populated_confusion_matrix = True

        return self.confusion_matrix_cache
