# CESI A4 Deep Learning Project Instructions

## Project Overview

Educational implementation of a custom neural network framework built from scratch (no external ML frameworks). Includes:

- Core layer and network orchestration with forward/backward propagation
- Plugin architecture for activation functions, loss functions, and training callbacks
- Hyperparameter grid search with comprehensive evaluation metrics
- Dataset utilities for binary health indicators classification
- Jupyter notebooks (`Livrable 1/2/3.ipynb`) demonstrating workflows and experiments

**Type**: Educational deep learning project  
**Language**: Python 3  
**Primary Artifacts**: Library code in `lib/`, experiment notebooks in root

---

## Quick Start Commands

### Test Suite

```bash
pytest .                                # Run all tests
pytest lib/neural_network/              # Run neural network tests
pytest lib/neural_network/loss/         # Run loss function tests
pytest path/to/test_file.py             # Run specific test file
```

**Test Configuration**: Pytest configured in `.vscode/settings.json` with unittest disabled. [View test files](../lib/neural_network/) — 7 total:

- Grid search, loss functions (BCE, CCE, MSE, MAE), callbacks (early stopping, visualization)

### Dependencies

Install from [requirements.txt](../requirements.txt): matplotlib, seaborn, pandas, numpy, ipywidgets, ipykernel, ipython.

---

## Architecture & Conventions

### Module Structure

```
lib/neural_network/                 # Core framework
├── neural_network.py                # NeuralNetwork orchestrator + History TypedDict
├── layer.py                         # Layer (neurons, weights, forward/backward)
├── grid_search.py                   # GridSearch with Params/Result TypedDicts
├── evaluation.py                    # Metrics: accuracy, precision, recall, F1, AUC, ROC
├── activation/                      # Plugin: Relu, Sigmoid, Tanh (+ base.py)
├── loss/                            # Plugin: BCE, CCE, MSE, MAE (+ base.py, tests)
└── callback/                        # Plugin: EarlyStopping, DrawRealTimeLoss (+ base.py, tests)

lib/dataset/                        # Data handling
├── dataset.py                       # Dataset class (exploration, cleaning, filtering)
└── display.py                       # Visualization utilities
```

### Key Design Patterns

1. **Strategy Pattern** (`activation/`, `loss/`, `callback/`)
    - All plugins inherit from abstract base class with `@abstractmethod`
    - Implementations follow naming: `Relu`, `BinaryCrossEntropy`, `EarlyStopping`
    - See [activation/base.py](../lib/neural_network/activation/base.py) as template

2. **Type-Safe Parameters** (TypedDict)
    - `History`: losses, validation losses, predictions, training data per epoch
    - `Params`: learning_rate, batch_size, epochs, loss, architecture config
    - `LayerParams`: neurons, dropout_rate, activation settings
    - `Result`: hyperparameter search results with scores and histories
    - See [grid_search.py](../lib/neural_network/grid_search.py) for TypedDict definitions

3. **Training Pipeline**
    - Callbacks inspect `History` to halt training (EarlyStopping)
    - Real-time visualization (DrawRealTimeLoss) updates during training
    - Comprehensive eval metrics in [evaluation.py](../lib/neural_network/evaluation.py)

### Code Conventions

| Aspect                | Rule                                                                                            | Example                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Class Names**       | PascalCase                                                                                      | `NeuralNetwork`, `Relu`, `BinaryCrossEntropy`                                     |
| **Functions/Methods** | snake_case                                                                                      | `compute()`, `forward()`, `backward()`, `derivative()`                            |
| **Type Hints**        | Full annotations; use `from __future__ import annotations`                                      | `def forward(self, x: np.ndarray) -> np.ndarray:`                                 |
| **Docstrings**        | Module + class + method-level; include math formulas                                            | See [binary_cross_entropy.py](../lib/neural_network/loss/binary_cross_entropy.py) |
| **Tests**             | Function-level (pytest), no unittest classes                                                    | `def test_binary_cross_entropy_compute():`                                        |
| **Abstraction**       | ABC + @abstractmethod for plugin systems                                                        | Copy [loss/base.py](../lib/neural_network/loss/base.py) pattern                   |
| **Pylint Rules**      | pyproject.toml disables: R0902 (too many attrs), R0913 (too many args), R0914 (too many locals) | See [pyproject.toml](../pyproject.toml)                                           |

---

## Development Workflows

### Adding a New Activation Function

1. Create file: `lib/neural_network/activation/my_activation.py`
2. Copy template from [activation/relu.py](../lib/neural_network/activation/relu.py)
3. Implement `forward()` and `derivative()` methods
4. Update [activation/**init**.py](../lib/neural_network/activation/__init__.py) with export
5. Create test: `lib/neural_network/activation/test_my_activation.py` following existing patterns

### Adding a Loss Function

1. Create file: `lib/neural_network/loss/my_loss.py`
2. Inherit from `LossBase` defined in [loss/base.py](../lib/neural_network/loss/base.py)
3. Implement `compute()` and `derivative()` with numerical stability
4. Add comprehensive docstring with math formula
5. Create test file with edge cases (e.g., log(0) prevention)
6. Update [loss/**init**.py](../lib/neural_network/loss/__init__.py)

### Running Experiments in Notebooks

- [Livrable 1.ipynb](../Livrable%201.ipynb) shows full end-to-end workflow (82 cells)
- Import: `from lib.neural_network import NeuralNetwork, GridSearch, Evaluation`
- Use GridSearch for hyperparameter tuning with `Params` TypedDict
- Visualize with callbacks and evaluation metrics

### Testing New Components

```bash
# Run all tests
pytest .

# Run with verbose output
pytest -v lib/neural_network/

# Run specific test
pytest lib/neural_network/test_grid_search.py::test_grid_search_params
```

---

## Common Pitfalls

1. **Edge Cases in Loss/Activation**: Handle log(0), log(1), division by zero
    - See test files (e.g., [test_binary_cross_entropy.py](../lib/neural_network/loss/test_binary_cross_entropy.py)) for patterns
2. **Type Hints Required**: All functions/methods must have type annotations
    - Use `from __future__ import annotations` at top of file
3. **Abstract Methods**: Plugin systems must inherit from base class
    - Forgetting `@abstractmethod` breaks contract

4. **Test Location**: Tests live in same directory as code (`test_*.py` files)
    - Named `test_<component>.py` for consistency

5. **Numerical Precision**: Use numpy operations; be mindful of float32 vs float64 in tests

---

## File Structure Reference

| File                   | Purpose                                                    | Exemplar                                         |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| **neural_network.py**  | Core NeuralNetwork class, training loop, History tracking  | [Link](../lib/neural_network/neural_network.py)  |
| **layer.py**           | Single layer with neurons, weights, forward/backward       | [Link](../lib/neural_network/layer.py)           |
| **grid_search.py**     | Hyperparameter tuning with TypedDict structures            | [Link](../lib/neural_network/grid_search.py)     |
| **evaluation.py**      | Metrics (accuracy, precision, recall, F1, AUC) + ROC curve | [Link](../lib/neural_network/evaluation.py)      |
| **loss/base.py**       | Abstract base for loss functions                           | [Link](../lib/neural_network/loss/base.py)       |
| **activation/relu.py** | Exemplary activation function implementation               | [Link](../lib/neural_network/activation/relu.py) |
| **callback/base.py**   | Abstract base for training callbacks                       | [Link](../lib/neural_network/callback/base.py)   |
| **dataset.py**         | Dataset exploration, cleaning, filtering utilities         | [Link](../lib/dataset/dataset.py)                |

---

## Jupyter Notebook Resources

- [Livrable 1.ipynb](../Livrable%201.ipynb) — Complete workflow example (import, train, evaluate, visualize)
- [Livrable 2.ipynb](../Livrable%202.ipynb) — Additional experiments
- [Livrable 3.ipynb](../Livrable%203.ipynb) — Final deliverable notebook

All notebooks import from `lib` package; check them for usage examples.

---

## Questions?

Refer to test files for edge case handling and usage patterns. Jupyter notebooks demonstrate end-to-end workflows. All classes and methods include docstrings with mathematical context.
