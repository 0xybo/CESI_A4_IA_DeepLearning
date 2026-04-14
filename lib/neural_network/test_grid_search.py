"""
Tests for the grid search functionality.

This module contains tests for the GridSearch class, which performs hyperparameter tuning for
neural networks. The tests include generating combinations of hyperparameters and running a grid
search on a simple dataset.
"""

import warnings

import numpy as np

from .grid_search import GridSearch, Params
from .activation.relu import Relu
from .activation.sigmoid import Sigmoid
from .loss.mean_squared_error import MeanSquaredError


def test_generate_combinations():
    """
    Test the generation of hyperparameter combinations.
    """
    grid_search = GridSearch()
    params: Params = {
        "learning_rate": [0.01, 0.001],
        "batch_size": [32, 64],
        "epochs": [10, 20],
        "loss": [MeanSquaredError()],
        "early_stopping_patience": [5],
        "architecture": [
            [
                {"neurons": [10], "dropout_rate": [0.2], "activation": [Relu()]},
                {"neurons": [1], "dropout_rate": [0.2], "activation": [Sigmoid()]},
            ]
        ],
    }

    combinations = grid_search.generate_combinations(params)  # type: ignore # pylint: disable=protected-access

    print("Generated combinations:")
    for i, combination in enumerate(combinations):
        print(f"  {i + 1}. {combination}")

    # 2 learning rates * 2 batch sizes * 2 epochs *
    # 1 loss * 1 early stopping patience * 1 architecture
    assert len(combinations) == 8


def test_grid_search():
    """
    Test the grid search functionality on a simple dataset.
    """
    x_train = np.random.rand(100, 5)
    y_train = (np.random.rand(100, 1) > 0.5).astype(int)
    x_val = np.random.rand(20, 5)
    y_val = (np.random.rand(20, 1) > 0.5).astype(int)
    results = GridSearch().search(
        {
            "learning_rate": [0.01],
            "batch_size": [32],
            "epochs": [10],
            "loss": [MeanSquaredError()],
            "early_stopping_patience": [5],
            "architecture": [
                [
                    {
                        "neurons": [10],
                        "dropout_rate": [0.2],
                        "activation": [Relu()],
                    },
                    {
                        "neurons": [1],
                        "dropout_rate": [0.2],
                        "activation": [Sigmoid()],
                    },
                ]
            ],
        },
        x_train,
        y_train,
        x_val,
        y_val,
    )

    metrics = results[0]["metrics"]

    # assert metrics["Accuracy"] == 0.5
    # assert metrics["Precision"] == 0.5
    # assert metrics["Recall"] == 1.0
    # assert metrics["F1 Score"] == 2 / 3

    print("Metrics:", metrics)


def test_grid_search_with_compare():
    """
    Test the grid search functionality with a graph comparison.
    """

    x_train = np.random.rand(100, 5)
    y_train = (np.random.rand(100, 1) > 0.5).astype(int)
    x_val = np.random.rand(20, 5)
    y_val = (np.random.rand(20, 1) > 0.5).astype(int)
    GridSearch().search_and_compare(
        {
            "learning_rate": [0.01, 0.001],
            "batch_size": [32, 64],
            "epochs": [10],
            "loss": [MeanSquaredError()],
            "early_stopping_patience": [5],
            "architecture": [
                [
                    {"neurons": [5], "dropout_rate": [0.2], "activation": [Relu()]},
                    {
                        "neurons": [10],
                        "dropout_rate": [0.2],
                        "activation": [Relu()],
                    },
                    {
                        "neurons": [1],
                        "dropout_rate": [0.2],
                        "activation": [Sigmoid()],
                    },
                ]
            ],
        },
        x_train,
        y_train,
        x_val,
        y_val,
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # Ignore warnings for cleaner test output
    # np.random.seed(42)  # For reproducibility
    test_generate_combinations()
    test_grid_search()
    test_grid_search_with_compare()

    print("Test passed!")
