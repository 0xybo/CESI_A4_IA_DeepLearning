# CESI A4 - Deep Learning Project

<div align="center">
  <h3>Implémentation d'un Framework de Réseau de Neurones à partir de Zéro</h3>
  <p><strong>Thomas VINET</strong> • <strong>Hugo HELM</strong> • <strong>Alban GODIER</strong></p>
  <p><em>CESI A4 - Module IA - Deep Learning</em></p>
</div>

---

## Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Livrables](#livrables)
- [Architecture du Framework](#architecture-du-framework)
- [Guide d'Utilisation](#guide-dutilisation)
- [Tests](#tests)

---

## Vue d'Ensemble

Ce projet implémente un **framework de réseau de neurones personnalisé** construit entièrement à partir de zéro, sans dépendre de frameworks ML externes (TensorFlow, PyTorch, etc.). Le projet démontre une compréhension approfondie des concepts fondamentaux du deep learning à travers trois livrables progressifs.

### Objectif Principal

Construire un classificateur binaire pour prédire les indicateurs de diabète à partir d'un dataset de données de santé BRFSS 2015 (253 680 observations).

### Caractéristiques Clés

[✓] **Réseau de neurones personnalisé** avec propagation avant/arrière  
[✓] **Architecture modulaire** avec pattern Strategy pour activations, pertes et callbacks  
[✓] **Grid Search** pour optimisation d'hyperparamètres  
[✓] **Multiples métriques d'évaluation** : Accuracy, Precision, Recall, F1, AUC, ROC  
[✓] **Callbacks avancés** : Early Stopping, Visualisation en temps réel, Adaptation du learning rate  
[✓] **IA Explicable** : Implémentations LIME et SHAP  
[✓] **MLOps** : Suivi des émissions carbone, barres de progression

---

## Structure du Projet

```
CESI_A4_IA_DeepLearning/
├── Livrable 1.ipynb              # Prétraitement des données
├── Livrable 2.ipynb              # Construction du réseau de neurones
├── Livrable 3.ipynb              # Optimisation avancée et IA Explicable
│
├── lib/                           # Framework principal
│   ├── __init__.py
│   ├── dataset/                   # Gestion des données
│   │   ├── dataset.py             # Classe Dataset (exploration, nettoyage)
│   │   ├── display.py             # Utilitaires de visualisation
│   │   └── __init__.py
│   │
│   ├── neural_network/            # Framework du réseau de neurones
│   │   ├── neural_network.py      # Orchestrateur principal + History TypedDict
│   │   ├── layer.py               # Couche (neurones, poids, forward/backward)
│   │   ├── grid_search.py         # Grid Search avec TypedDicts Params/Result
│   │   ├── evaluation.py          # Métriques (accuracy, precision, recall, F1, AUC, ROC)
│   │   │
│   │   ├── activation/            # Plugin d'activations
│   │   │   ├── base.py            # Classe abstraite
│   │   │   ├── relu.py            # ReLU
│   │   │   ├── sigmoid.py         # Sigmoid
│   │   │   ├── tanh.py            # Tanh
│   │   │   ├── none.py            # Linear (pas d'activation)
│   │   │   └── __init__.py
│   │   │
│   │   ├── loss/                  # Plugin de fonctions de perte
│   │   │   ├── base.py            # Classe abstraite
│   │   │   ├── binary_cross_entropy.py
│   │   │   ├── binary_cross_entropy_sigmoid.py
│   │   │   ├── categorical_cross_entropy.py
│   │   │   ├── mean_squared_error.py
│   │   │   ├── mean_absolute_error.py
│   │   │   ├── test_*.py          # Tests unitaires
│   │   │   └── __init__.py
│   │   │
│   │   ├── callback/              # Plugin de callbacks
│   │   │   ├── base.py            # Classe abstraite
│   │   │   ├── early_stopping.py  # Arrêt précoce
│   │   │   ├── draw_real_time_loss.py  # Visualisation en temps réel
│   │   │   ├── train_progress_bar.py   # Barre de progression
│   │   │   ├── epoch_progress_bar.py   # Barre par époque
│   │   │   ├── carbon_emissions.py     # Suivi des émissions CO2
│   │   │   ├── adaptive_learning_rate_*.py  # Adaptation du learning rate
│   │   │   ├── test_*.py          # Tests unitaires
│   │   │   └── __init__.py
│   │   │
│   │   ├── explainatinator/       # IA Explicable
│   │   │   ├── base.py            # Classe abstraite
│   │   │   ├── lime.py            # LIME (Local Interpretable Model-agnostic Explanations)
│   │   │   ├── shap.py            # SHAP (SHapley Additive exPlanations)
│   │   │   └── __init__.py
│   │   │
│   │   ├── test_*.py              # Tests principaux
│   │   └── __init__.py
│   │
│   └── utils/                     # Utilitaires
│       └── run_coroutine_sync.py
│
├── src/                           # Code source supplémentaire
│   └── grid_search.py
│
├── dataset/                       # Données
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv  # Dataset brut
│   ├── dataset_train.csv          # Données d'entraînement (90%)
│   └── dataset_validation.csv     # Données de validation (10%)
│
├── grid_search_results/           # Résultats des recherches en grille
│   └── *.json                     # Historiques d'optimisation
│
├── assets/                        # Ressources (logos, diagrammes)
│   └── cesi.png
│
├── diagrams/                      # Diagrammes
│   └── neural_network.mmd         # Diagramme Mermaid du réseau
│
├── pyproject.toml                 # Configuration Pylint
├── requirements.txt               # Dépendances
└── README.md                      # Ce fichier
```

---

## Installation

### Prérequis

- Python 3.8+
- pip

### Étapes

1. **Cloner ou télécharger le projet**

    ```bash
    cd CESI_A4_IA_DeepLearning
    ```

2. **Créer un environnement virtuel (optionnel mais recommandé)**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3. **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

### Dépendances Principales

| Package    | Version | Utilité                   |
| ---------- | ------- | ------------------------- |
| numpy      | 2.4.2   | Calculs numériques        |
| pandas     | 3.0.1   | Manipulation de données   |
| matplotlib | 3.10.8  | Visualisation             |
| seaborn    | 0.13.2  | Visualisation statistique |
| ipywidgets | 8.1.8   | Widgets Jupyter           |
| CodeCarbon | 3.2.6   | Suivi émissions carbone   |
| tqdm       | 4.67.3  | Barres de progression     |

---

## Livrables

### Livrable 1 - Prétraitement des Données (82 cellules)

**Fichier**: [`Livrable 1.ipynb`](Livrable%201.ipynb)

**Objectifs**:

1. [✓] Chargement et compréhension du dataset
2. [✓] Séparation variables cible / explicatives
3. [✓] Scission train/validation (90% / 10%)
4. [✓] Typage des variables (qualitatives/quantitatives)
5. [✓] Nettoyage des doublons
6. [✓] Gestion des valeurs manquantes
7. [✓] Analyse exploratoire quantitative et qualitative (EDA)
8. [✓] Détection des valeurs aberrantes (Méthode IQR)
9. [✓] Traitement des outliers (Trimming)
10. [✓] Normalisation des données (Min-Max Scaling)
11. [✓] Analyse des corrélations (Pearson, Spearman, Kendall)

**Résultats Clés**:

- 253 680 observations initiales → 247 221 après traitement (2.46% outliers)
- 21 variables explicatives (16 retenues)
- Distribution équilibrée entre classes (35% diabétiques, 65% non-diabétiques)
- Corrélation identifiée pour: HighBP, BMI, DiffWalk, HighChol, Age, HeartDiseaseorAttack

---

### Livrable 2 - Construction du Réseau de Neurones

**Fichier**: [`Livrable 2.ipynb`](Livrable%202.ipynb)

**Objectifs**:

1. [✓] Démonstrations mathématiques (descente de gradient, backpropagation)
2. [✓] Implémentation de la propagation avant
3. [✓] Implémentation de la rétropropagation
4. [✓] Entraînement du réseau
5. [✓] Évaluation sur données de validation
6. [✓] Visualisation des pertes
7. [✓] Étude d'impact des hyperparamètres

**Architecture du Modèle Baseline**:

```
Input Layer (16 neurones)
    ↓
Hidden Layer 1 (64 neurones, ReLU, Dropout 0.2)
    ↓
Hidden Layer 2 (32 neurones, ReLU, Dropout 0.2)
    ↓
Output Layer (1 neurone, Sigmoid)
    ↓
Binary Classification (Diabète: Oui/Non)
```

**Fonctions Utilisées**:

- **Loss**: Mean Squared Error (MSE) ou Binary Cross-Entropy
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Optimiseur**: Gradient Descent
- **Callbacks**: Early Stopping, Visualisation temps réel

---

### Livrable 3 - Optimisation Avancée et IA Explicable

**Fichier**: [`Livrable 3.ipynb`](Livrable%203.ipynb)

**Objectifs**:

1. [✓] Grid Search pour optimisation d'hyperparamètres
2. [✓] Comparaison de multiples architectures
3. [✓] Adaptation du learning rate (Step Decay, Reduce on Plateau)
4. [✓] Suivi des émissions carbone (CodeCarbon)
5. [✓] IA Explicable (LIME et SHAP)
6. [✓] Analyse d'importances des features
7. [✓] Visualisation des décisions du modèle

**Techniques MLOps Implémentées**:

| Technique                  | Classe                                 | Utilité                                  |
| -------------------------- | -------------------------------------- | ---------------------------------------- |
| **Early Stopping**         | `EarlyStopping`                        | Arrête si validation loss n'améliore pas |
| **Adaptive Learning Rate** | `StepDecay`, `ReduceOnPlateau`         | Réduit LR si plateau atteint             |
| **Carbon Tracking**        | `CarbonEmissions`                      | Estime l'impact CO₂ de l'entraînement    |
| **Progress Monitoring**    | `TrainProgressBar`, `EpochProgressBar` | Suivi visuel de l'entraînement           |
| **LIME**                   | `LIME`                                 | Explique prédictions localement          |
| **SHAP**                   | `SHAP`                                 | Attribue importance aux features         |

---

## Architecture du Framework

### Pattern de Conception

Le framework utilise le **Strategy Pattern** pour permettre l'extensibilité :

#### 1. Activations (`lib/neural_network/activation/`)

```python
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray: pass
```

**Implémentations**:

- `Relu`: $f(x) = \max(0, x)$
- `Sigmoid`: $f(x) = \frac{1}{1 + e^{-x}}$
- `Tanh`: $f(x) = \tanh(x)$
- `None`: Linéaire (pas d'activation)

#### 2. Fonctions de Perte (`lib/neural_network/loss/`)

```python
class LossFunction(ABC):
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: pass
```

**Implémentations**:

- `MeanSquaredError`: $MSE = \frac{1}{n}\sum(y - \hat{y})^2$
- `BinaryCrossEntropy`: $BCE = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$
- `CategoricalCrossEntropy`: Pour classification multi-classe
- `MeanAbsoluteError`: $MAE = \frac{1}{n}\sum|y - \hat{y}|$

#### 3. Callbacks (`lib/neural_network/callback/`)

```python
class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, history: History) -> bool: pass
```

### Structures TypedDict

Type-safe parameter passing:

```python
class Params(TypedDict):
    learning_rate: float
    batch_size: int
    epochs: int
    loss: LossFunction
    activation: ActivationFunction
    layers: List[LayerParams]

class History(TypedDict):
    losses: List[float]
    validation_losses: List[float]
    predictions: np.ndarray
    training_data: np.ndarray
```

---

## Guide d'Utilisation

### Exemple Basique

```python
from lib.neural_network import NeuralNetwork, Layer, Sigmoid, Relu
from lib.neural_network.loss import BinaryCrossEntropy
from lib.neural_network.callback import EarlyStopping
import numpy as np

# Créer un réseau
network = NeuralNetwork([
    Layer(neurons=64, activation=Relu(), dropout_rate=0.2),
    Layer(neurons=32, activation=Relu(), dropout_rate=0.2),
    Layer(neurons=1, activation=Sigmoid()),
], loss=BinaryCrossEntropy(), inputs=16)

# Ajouter des callbacks
network.add_callback(EarlyStopping(patience=10))

# Entraîner
network.fit(
    X_train,           # np.ndarray [n_samples, n_features]
    y_train,           # np.ndarray [n_samples,]
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

# Prédire
predictions = network.predict(X_test)
```

### Avec Grid Search

```python
from lib.neural_network.grid_search import GridSearch, Params

params_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 200],
    'layers': [
        [Layer(32, Relu()), Layer(16, Relu()), Layer(1, Sigmoid())],
        [Layer(64, Relu()), Layer(32, Relu()), Layer(1, Sigmoid())],
    ]
}

grid_search = GridSearch(params_grid)
results = grid_search.search(X_train, y_train, X_val, y_val)

# Récupérer le meilleur modèle
best_model = results.best_model
print(f"Best accuracy: {results.best_score}")
```

### Avec Dataset Helper

```python
from lib.dataset import Dataset

# Charger et explorer
dataset = Dataset.from_csv('dataset.csv', target='Diabetes_binary')

# Analyse
dataset.info()
dataset.describe()
dataset.get_missing_values()
dataset.get_outliers()

# Prétraitement
dataset.drop(['col1', 'col2'], inplace=True)
normalized = dataset.normalize_minmax(['BMI'])
filtered = dataset.filter_outliers_iqr(column='BMI')

# Visualisation
dataset.draw_distributions()
dataset.draw_correlations_with_target(method='spearman')

# Exporter
train, val = dataset.export_to_csv('./data', train_ratio=0.9)
```

---

## Tests

### Exécuter les Tests

```bash
# Tous les tests
pytest .

# Tests spécifiques
pytest lib/neural_network/loss/test_binary_cross_entropy.py
pytest lib/neural_network/ -v

# Avec couverture
pytest --cov=lib
```

### Structure des Tests

Les tests sont organisés par module avec **7 fichiers de test principaux** :

| Fichier                                  | Couverture               |
| ---------------------------------------- | ------------------------ |
| `test_grid_search.py`                    | Grid Search, Params      |
| `test_neural_network.py`                 | Réseau, forward/backward |
| `loss/test_binary_cross_entropy.py`      | BCE (edge cases)         |
| `loss/test_categorical_cross_entropy.py` | CCE                      |
| `loss/test_mean_squared_error.py`        | MSE                      |
| `loss/test_mean_absolute_error.py`       | MAE                      |
| `callback/test_early_stopping.py`        | Early Stopping           |
| `callback/test_draw_real_time_loss.py`   | Visualisation            |

**Note**: Tests configurés dans `.vscode/settings.json` avec unittest désactivé en faveur de pytest.

---

## Résultats et Métriques

### Performances du Modèle

Les livrables incluent l'évaluation complète avec métriques calculées dans [`lib/neural_network/evaluation.py`](lib/neural_network/evaluation.py):

| Métrique      | Description                          | Formule                             |
| ------------- | ------------------------------------ | ----------------------------------- |
| **Accuracy**  | % de prédictions correctes           | $\frac{TP+TN}{TP+TN+FP+FN}$         |
| **Precision** | % de prédictions positives correctes | $\frac{TP}{TP+FP}$                  |
| **Recall**    | % de positifs détectés               | $\frac{TP}{TP+FN}$                  |
| **F1-Score**  | Moyenne harmonique Precision/Recall  | $2 \times \frac{P \times R}{P + R}$ |
| **AUC**       | Area Under ROC Curve                 | (0 à 1)                             |
| **ROC Curve** | Graphique TP vs FP rates             | Courbe d'évaluation                 |

### Visualisations Incluses

- Histogrammes de distributions
- Matrices de corrélation
- Courbes d'apprentissage (loss vs epochs)
- Courbes ROC
- Matrices de confusion
- Graphiques LIME/SHAP

---

## Conventions de Code

| Aspect         | Convention                           | Exemple                                                                                                   |
| -------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Classes**    | PascalCase                           | `NeuralNetwork`, `Relu`                                                                                   |
| **Fonctions**  | snake_case                           | `forward()`, `backward()`                                                                                 |
| **Type Hints** | Complètes                            | `def fit(self, X: np.ndarray) -> None:`                                                                   |
| **Docstrings** | Module + Classe + Méthode            | Voir [`lib/neural_network/loss/binary_cross_entropy.py`](lib/neural_network/loss/binary_cross_entropy.py) |
| **Imports**    | `from __future__ import annotations` | En haut de chaque fichier                                                                                 |
| **Tests**      | `test_<module>.py`                   | Dans le même répertoire                                                                                   |
| **Pylint**     | Désactivisé: R0902, R0913, R0914     | Cf. [`pyproject.toml`](pyproject.toml)                                                                    |

---

## Dépannage

### Problèmes Courants

**Q: Import de `lib` échoue**

- [✓] Vérifiez que vous êtes dans le bon répertoire
- [✓] Vérifiez que `__init__.py` existe dans chaque dossier

**Q: Tests ne trouvent pas les modules**

- [✓] Exécutez `pytest .` depuis la racine du projet
- [✓] Assurez-vous que pytest est installé

**Q: Numpy/Pandas warnings**

- [✓] Les warnings sont supprimés dans les notebooks avec `warnings.filterwarnings("ignore")`

---

## Notes Pédagogiques

Ce projet offre une implémentation éducative complète permettant de comprendre:

1. **Mathématiques du Deep Learning**
    - Propagation avant et rétropropagation
    - Descente de gradient
    - Fonctions d'activation et pertes
    - Optimisation d'hyperparamètres

2. **Génie Logiciel**
    - Pattern Strategy et abstractions
    - Type hints pour la sécurité
    - Architecture modulaire extensible
    - Tests unitaires et intégration

3. **MLOps et Production**
    - Grid Search systématique
    - Callbacks et monitoring
    - Suivi des émissions carbone
    - IA Explicable (LIME/SHAP)

4. **Analyse de Données**
    - EDA complète
    - Détection d'anomalies
    - Normalisation et prétraitement
    - Analyse de corrélations

---

## Auteurs

- **Thomas VINET**
- **Hugo HELM**
- **Alban GODIER**

**Formation**: CESI A4 - Cycle 4  
**Module**: IA - Deep Learning  
**Date**: 2026

---

---

## Références

- [BRFSS 2015 Dataset Documentation](https://www.cdc.gov/brfss/annual_data/2024/pdf/2024-calculated-variables-version4-508.pdf)
- [Deep Learning - Goodfellow et al.](http://deeplearningbook.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [LIME: "Why Should I Trust You?"](https://arxiv.org/abs/1602.04938)
- [SHAP: Additive Feature Attribution Methods](https://arxiv.org/abs/1705.07874)
