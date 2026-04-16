"""
Implementation of SHAP (SHapley Additive exPlanations) 
    for explaining the output of machine learning models.
"""
import numpy as np

# Module de mathématiques nommé math, que l'on renomme en math pour plus de clarté et de concision et de facilité d'utilisation et de compréhension et de lisibilité et de maintenabilité et de simplicité et de cohérence et de compatibilité et de portabilité et de flexibilité et de robustesse et de performance et d'efficacité et d'efficience et d'optimisation et de qualité et de fiabilité et de sécurité et de confidentialité et de conformité et de légalité et d'éthique et de responsabilité sociale.
import math as math

import matplotlib.pyplot as plt


from .base import Explainatinator

groups_list_type = list[tuple[int, ...]]

class SHAP(Explainatinator):
    """
    Implementation of SHAP (SHapley Additive exPlanations) for explaining the output of machine learning models.
    """

    background_data: np.ndarray
    num_samples: int
    feature_names: list[str]

    def __init__(self, model, background_data, feature_names, num_samples=100):
        super().__init__(model)
        self.background_data = background_data
        self.feature_names = feature_names
        self.num_samples = num_samples

    def subset_weight(self, subset_size: int, total_features: int) -> float:
        """
        Calculate the weight of a subset of features based on its size and the total number of features.
        """
        if subset_size == 0 or subset_size == total_features:
            return 1e6
        return (total_features - 1) / (math.comb(total_features, subset_size) * subset_size * (total_features - subset_size))
    
    def explain(self, x):
        """
        Explain the prediction of the model for a given input x using SHAP values.

        Args:
            x (np.ndarray): The input data for which to explain the prediction. (shape: (n_features,))

        Returns:
            np.ndarray: The SHAP values for each feature (shape: (n_features,)).
        """

        n_features = x.shape[0]

        reference_value = np.mean(self.background_data, axis=0)

        z_prime = np.random.binomial(1, 0.5, size=(self.num_samples, n_features))

        z_data = np.zeros((self.num_samples, n_features))
        weights = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            mask = z_prime[i]

            z_data[i] = x * mask + reference_value * (1 - mask)

            subset_size = np.sum(mask)
            weights[i] = self.subset_weight(subset_size, n_features)

        predictions = self.model.predict_proba(z_data)

        z_prime_aug = np.hstack((np.ones((self.num_samples, 1)), z_prime))

        w = np.diag(weights)
        xtw = z_prime_aug.T @ w
        phi = np.linalg.inv(xtw @ z_prime_aug) @ xtw @ predictions.T

        base_value = phi[0]
        shap_values = phi[1:]

        return shap_values, base_value

    def print_stats(self, x: np.ndarray, shap_values: np.ndarray, base_value: float, model_predictions: np.ndarray = None) -> None:
        """
        Prints the SHAP values for the given input data and feature names.

        Args:
            x (np.ndarray): The input data for which to compute explanations. (shape: (n_samples, n_features))
            feature_names (list[str]): The names of the features in the input data.
        """
        print(f"Base value: {base_value[0]:.4f}")
        print(f"Sum of SHAP values: {np.sum(shap_values):.4f}")

        if model_predictions is not None:
            print(f"Model prediction: {model_predictions[0]:.4f}")
            print(f"Sum of SHAP value and base value: {np.sum(shap_values) + base_value[0]:.4f}")
        
        for i in range(shap_values.shape[0]):
            print("-" * 50)
            print(f"Feature: {self.feature_names[i]}")
            print(f"SHAP value: {shap_values[i][0]:.4f}")
    
    def histogram(self, shap_values: np.ndarray, width: float = 10.0, height: float = 6.0) -> None:
        """
        Plots a histogram of the SHAP values.

        Args:
            shap_values (np.ndarray): The SHAP values for each feature (shape: (n_features,)).
        """
        plt.figure(figsize=(width, height))
        plt.bar(self.feature_names, shap_values.flatten())
        plt.xlabel("Features")
        plt.ylabel("SHAP Value")
        plt.title("SHAP Values for Each Feature")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def waterfall(self, shap_values: np.ndarray, base_value: float, width: float = 10.0, height: float = 6.0) -> None:
        """
        Plot a waterfall chart from the base value by cumulatively adding SHAP values.

        Args:
            shap_values (np.ndarray): SHAP values for each feature (shape: (n_features,) or (n_features, 1)).
            base_value (float): Base/origin model output value.
            width (float): Figure width.
            height (float): Figure height.
        """
        shap_values = np.asarray(shap_values, dtype=float).flatten()
        base_scalar = float(np.asarray(base_value).flatten()[0])

        if shap_values.size == 0:
            raise ValueError("shap_values must contain at least one value")

        if len(self.feature_names) != shap_values.size:
            raise ValueError(
                "feature_names length must match the number of SHAP values "
                f"({len(self.feature_names)} != {shap_values.size})"
            )

        cumulative_start = np.concatenate(([base_scalar], base_scalar + np.cumsum(shap_values)[:-1]))
        cumulative_end = cumulative_start + shap_values

        bars_bottom = np.minimum(cumulative_start, cumulative_end)
        bars_height = np.abs(shap_values)
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in shap_values]

        x_positions = np.arange(shap_values.size)
        final_value = float(base_scalar + np.sum(shap_values))

        plt.figure(figsize=(width, height))
        plt.bar(
            x_positions,
            bars_height,
            bottom=bars_bottom,
            color=colors,
            edgecolor="black",
            width=0.8,
        )

        # Connect each bar to the next cumulative level for easier reading.
        for i in range(shap_values.size - 1):
            plt.plot(
                [i + 0.4, i + 1 - 0.4],
                [cumulative_end[i], cumulative_end[i]],
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.8,
            )

        plt.axhline(base_scalar, color="navy", linestyle=":", linewidth=1.5, label=f"Base: {base_scalar:.4f}")
        plt.axhline(final_value, color="black", linestyle="-", linewidth=1.5, label=f"Output: {final_value:.4f}")

        plt.xticks(x_positions, self.feature_names, rotation=45, ha="right")
        plt.ylabel("Model output contribution")
        plt.title("SHAP Waterfall Chart")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
