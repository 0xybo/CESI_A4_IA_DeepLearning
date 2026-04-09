"""
Evaluation Class  for Neural Networks
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .neural_network import NeuralNetwork
from typing import Optional, Tuple

class Evaluation:
    """
    Evaluates the performance of a neural network on a validation dataset.
    This class provides methods to calculate various performance metrics such as accuracy, precision, recall, F1 score, and AUC. It also includes a method to draw the ROC curve.
        Attributes:
            • neural_network (NeuralNetwork): The neural network to evaluate.
            • x_validation (np.ndarray): The validation input data.
            • y_validation (np.ndarray): The validation target data.
            • _confusion_matrix (Tuple[int, int, int, int]): A tuple containing the counts of true positives, false positives, true negatives, and false negatives.
        Methods:
        • __init__(self, x_validation: Optional[np.ndarray] = None, y_validation: Optional[np.ndarray] = None) -> None: Initializes the Evaluation object with optional validation data.
        • validate(self, neural_network: NeuralNetwork) -> pd.DataFrame: Validates the neural network and returns a DataFrame containing the calculated metrics.
        • accuracy(self) -> float: Calculates and returns the accuracy of the neural network.
        • precision(self) -> float: Calculates and returns the precision of the neural network.
        • recall(self) -> float: Calculates and returns the recall of the neural network.
        • f1_score(self) -> float: Calculates and returns the F1 score of the neural network.
        • auc(self) -> float: Calculates and returns the AUC of the neural network.
        • draw_roc(self) -> Tuple[np.ndarray, np.ndarray]: Draws the ROC curve for the neural network.
        • confusion_matrix(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]: Calculates and returns the confusion matrix for the neural network.
    """
    
    neural_network: NeuralNetwork
    x_validation: np.ndarray
    y_validation: np.ndarray
    _confusion_matrix: Tuple[int, int, int, int] # (TP, FP, TN, FN)

    def __init__(self, x_validation: Optional[np.ndarray] = None, y_validation: Optional[np.ndarray] = None) -> None: 
        """
        Initializes the Evaluation object with optional validation data.
        Args:            
            x_validation (Optional[np.ndarray]): The validation input data. Default is None.
            y_validation (Optional[np.ndarray]): The validation target data. Default is None.
        """
        self.x_validation = x_validation
        self.y_validation = y_validation
        self._confusion_matrix = None
        
    def validate(self, neural_network: NeuralNetwork) -> Dict: 
        """
        Validates the neural network and returns a DataFrame containing the calculated metrics.
        Args:
            neural_network (NeuralNetwork): The neural network to validate.
        Returns:
            pd.DataFrame: A DataFrame containing the calculated metrics (accuracy, precision, recall, F1 score, and AUC).
        """
        self.neural_network = neural_network
        self.confusion_matrix()  # Calculate confusion matrix to populate it

        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1 Score': self.f1_score(),
            # 'AUC': self.auc()
        }
        
        return metrics
   
    def accuracy(self) -> float:
        """
        Calculates and returns the accuracy of the neural network.
        Returns:
            float: The accuracy value.
        """
        true_positives: int = self._confusion_matrix[0]
        false_positives: int = self._confusion_matrix[1]
        true_negatives: int = self._confusion_matrix[2]
        false_negatives: int = self._confusion_matrix[3]

        return (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

    def precision(self) -> float:
        """
        Calculates and returns the precision of the neural network.
        Returns:
            float: The precision value.
        """
        true_positives: int = self._confusion_matrix[0]
        false_positives: int = self._confusion_matrix[1]

        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
    def recall(self) -> float: 
        """
        Calculates and returns the recall of the neural network.
        Returns:
            float: The recall value.
        """
        true_positives: int = self._confusion_matrix[0]
        false_negatives: int = self._confusion_matrix[3]
        
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    def f1_score(self) -> float: 
        """Calculates and returns the F1 score of the neural network.
        Returns:
            float: The F1 score value.
        """
        precision: float = self.precision()
        recall: float = self.recall()

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def auc(self) -> float: 
        """
        Calculates and returns the AUC of the neural network.
        Returns:
            float: The AUC value.
        """
        true_positives_rate_list,false_positives_rate_list = self.draw_roc()
        auc_value = np.trapz(true_positives_rate_list, false_positives_rate_list)
        
        return auc_value
    
    def draw_roc(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draws the ROC curve for the neural network.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The true positive rate and false positive rate arrays.
        Actions:
            Plots the ROC curve using matplotlib.
        """
        y_reel= np.ndarray([])
        y_pred= np.ndarray([])
        history_list: np.ndarray = self.neural_network.history
        for echo in history_list:
            y_reel = np.append(y_reel, echo['y_train'])
            y_pred = np.append(y_pred, echo['y_pred'])

        indices_tries = np.argsort(y_pred)
        print('=' * 80)
        print('y_reel', y_reel.shape)
        print('y_pred', y_pred.shape)
        print('=' * 80)
        y_reel = y_reel[indices_tries]
        y_pred = y_pred[indices_tries]

        true_positives_rate_list: np.ndarray[float] = [0.0]
        false_positives_rate_list: np.ndarray[float] = [0.0]
        
        nb_positifs = np.sum(y_reel == 1)
        nb_negatifs = np.sum(y_reel == 0)
        
        true_positives = 0
        false_positives = 0
        
        for i in range(len(y_reel)):
            if y_reel[i] == 1:
                true_positives += 1
            else:
                false_positives += 1
            
            true_positives_rate_list = np.append(true_positives_rate_list, true_positives / nb_positifs)
            false_positives_rate_list = np.append(false_positives_rate_list, false_positives / nb_negatifs)
        
        plt.plot(false_positives_rate_list, true_positives_rate_list, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        
        return true_positives_rate_list,false_positives_rate_list

    def confusion_matrix(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
        """
        Calculates and returns the confusion matrix for the neural network.
        If x and y are not provided, it uses the validation data stored in the object.
        Args:
            x (Optional[np.ndarray]): The input data to use for calculating the confusion matrix. Default is None.
            y (Optional[np.ndarray]): The target data to use for calculating the confusion matrix. Default is None.
        """
        if self._confusion_matrix is None:
            if x is None:
                x = self.x_validation
            if y is None:
                y = self.y_validation
            
            if x is None or y is None:
                raise ValueError("Validation data not provided.")
            
            y_pred: np.ndarray = self.neural_network.predict(x)

            self._confusion_matrix = (
                np.sum((y_pred == 1) & (y == 1)),   # true positives
                np.sum((y_pred == 1) & (y == 0)),   # false positives
                np.sum((y_pred == 0) & (y == 0)),   # true negatives
                np.sum((y_pred == 0) & (y == 1))    # false negatives
            )

        print(self._confusion_matrix) 
        print(y_pred)
        return self._confusion_matrix