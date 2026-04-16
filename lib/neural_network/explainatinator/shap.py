"""
Implementation of SHAP (SHapley Additive exPlanations) 
    for explaining the output of machine learning models.
"""
import numpy as np
import math
from .base import Explainatinator

groups_list_type = list[tuple[int, ...]]

class SHAP(Explainatinator):
    """
    Class implementing SHAP (SHapley Additive exPlanations) for 
        explaining the output of machine learning models.
    """
    num_samples: int
    num_groups: int
    data : np.ndarray
    numpy_generator: np.random.Generator

    def __init__(self, model, data: np.ndarray, num_samples: int = 50, num_groups: int = None) -> None:
        """
        Initializes the SHAP explainatinator with the given model and parameters for generating groups of features.

        Args:
            model: The machine learning model for which to generate explanations.
            data: The training data used for generating explanations. This is necessary for computing the average values of features when generating the groups of features.
            num_samples (int, optional): The number of samples to use for generating explanations 
            num_groups (int, optional): The maximum number of groups to generate.
        """
        super().__init__(model)
        self.data = data

        self.num_groups = num_groups if num_groups is not None else 2 ** data.shape[1]
        self.num_samples = num_samples
        self.numpy_generator = np.random.default_rng(seed=42)



    def __sample_coalitions(self, n_features: int, target: int) -> groups_list_type:
        """
        Generates random coalitions of features for a given target feature
            ensuring that the target feature is not included in any coalition.

        Args:
            n_features (int): The total number of features in the input data.
            target (int): The target feature for which to generate coalitions.
        """
        features = np.array([i for i in range(n_features) if i != target], dtype=int)

        coalitions: groups_list_type = []
        for _ in range(self.num_groups):
            size = int(self.numpy_generator.integers(0, len(features) + 1))
            S = tuple(self.numpy_generator.choice(features, size=size, replace=False))
            coalitions.append(S)

        return coalitions
    
    def __weight(self, S_size: int, n_features: int) -> float:
        """
        Computes the weight of a coalition of features based on its size and the total number of features.

        Args:
            S_size (int): The size of the coalition of features.
            n_features (int): The total number of features in the input data.
        """
        return (
            math.factorial(S_size)
            * math.factorial(n_features - S_size - 1)
            / math.factorial(n_features)
        )
    
    def __estimate_v(self, x: np.ndarray, S: tuple[int, ...], target_class: int) -> float:
        """
        Estimates the value of a coalition of features by sampling from the training data and computing the average prediction of the model for the masked input.
        
        Args:
            x (np.ndarray): The input sample for which to compute the explanation.
            S (tuple): The coalition of features for which to estimate the value.
            target (int): The target feature for which to compute the explanation.
        """
        
        idx = self.numpy_generator.choice(len(self.data), size=self.num_samples, replace=True)
        masked = self.data[idx].copy()

        if S:
            s_idx = np.asarray(S, dtype=int)
            masked[:, s_idx] = x[s_idx]

        # Batch inference is significantly faster than one prediction per sample.
        preds = self.model.predict_proba(masked)
        return float(np.mean(preds[:, target_class]))

    def explain(self, x: np.ndarray, target_class: int = 1) -> np.ndarray:
        """
        Computes the SHAP values for the given input data and target class.

        Args:
            x (np.ndarray): The input data for which to compute explanations.
            target_class (int): The target class for which to compute explanations.

        Returns:
            np.ndarray: The SHAP values for the input data and target class.
        """
        n_samples, n_features = x.shape
        shap_values = np.zeros((n_samples, n_features))

        if target_class < 0:
            raise ValueError("target_class must be non-negative")

        for i_sample in range(n_samples):
            x_i = x[i_sample]

            for feature in range(n_features):
                coalitions = self.__sample_coalitions(n_features, feature)
                print(coalitions)
                print(len(coalitions))

                phi = 0.0

                for S in coalitions:
                    S_with_i = tuple(sorted(S + (feature,)))

                    v_S = self.__estimate_v(x_i, S, target_class)
                    v_S_i = self.__estimate_v(x_i, S_with_i, target_class)

                    w = self.__weight(len(S), n_features)

                    phi += w * (v_S_i - v_S)

                shap_values[i_sample, feature] = phi

        return shap_values