"""
Implementation of LIME (Local Interpretable Model-agnostic Explanations) 
    for explaining the output of machine learning models.
"""
import numpy as np
from .base import Explainatinator
from sklearn.linear_model import LogisticRegression

class LIME(Explainatinator):
    """
    Class implementing LIME (Local Interpretable Model-agnostic Explanations) 
        for explaining the output of machine learning models.
    """
    def explain(self, x: np.ndarray, num_perturbations: int = 1000, sigma: float = 0.1) -> np.ndarray:
        
        # Assurer que x est 2D (batch_size, features)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        y_pred = self.model.predict(x)

        generator = np.random.default_rng(42)
        # Créer perturbations: (num_perturbations, num_features)
        noise = generator.normal(0, sigma, size=(num_perturbations, x.shape[1]))
        
        # Ajouter le bruit à x pour créer les données perturbées
        perturbed_data = x + noise
        
        # Prédictions pour les données perturbées
        perturbed_preds = self.model.predict(perturbed_data).squeeze()
        
        # Calculer les distances par rapport à l'exemple original
        distances = np.linalg.norm(perturbed_preds - y_pred, axis=0)
        weights = np.exp(-(distances**2) / (sigma**2))
        
        # Entraîner un modèle linéaire local
        LR_model = LogisticRegression()
        LR_model.fit(perturbed_data, perturbed_preds, sample_weight=weights)
        
        return LR_model.coef_[0]