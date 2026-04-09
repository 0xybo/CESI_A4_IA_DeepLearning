import numpy as np

from .grid_search import GridSearch, LayerParams, Params, Result
from .activation.relu import Relu
from .activation.sigmoid import Sigmoid
from .activation.tanh import Tanh
from .loss.mean_squared_error import MeanSquaredError

import warnings


def test_generate_combinations():
    grid_search = GridSearch()
    params: Params = {
        "learning_rate": [0.01, 0.001],
        "batch_size": [32, 64],
        "epochs": [10, 20],
        "loss": [MeanSquaredError()],
        "early_stopping_patience": [5],
        "early_stopping_delta": [0.001],
        "architecture": [
            [
                {"neurons": [10], "dropout_rate": [0.2], "activation": [Relu()]},
                {"neurons": [10], "dropout_rate": [0.2], "activation": [Sigmoid()]},
            ]
        ],
    }
    combinations = grid_search._GridSearch__generate_combinations(params)  # type: ignore
    assert len(combinations) == 8


def test_grid_search():
    x_train = np.random.rand(100, 5)
    y_train = np.random.rand(100, 1)
    x_val = np.random.rand(20, 5)
    y_val = np.random.rand(20, 1)
    print(GridSearch().search(
        {
            "learning_rate": [0.2],
            "batch_size": [32],
            "epochs": [10],
            "loss": [MeanSquaredError()],
            "early_stopping_patience": [5],
            "architecture": [
                [
                    {"neurons": [5], "dropout_rate": [0.2], "activation": [Relu()]},
                    {"neurons": [10], "dropout_rate": [0.2], "activation": [Relu()]},
                    {"neurons": [1], "dropout_rate": [0.2], "activation": [Sigmoid()]},
                ]
            ],
        },
        x_train,
        y_train,
        x_val,
        y_val,
    )[0]['metrics'])


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # Ignore warnings for cleaner test output
    np.random.seed(42)  # For reproducibility
    test_generate_combinations()
    test_grid_search()

    print("Test passed!")
