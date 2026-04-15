import numpy as np
import pandas as pd
import json
import time

from lib.neural_network.activation.relu import Relu
from lib.neural_network.activation.sigmoid import Sigmoid
from lib.neural_network.loss.mean_squared_error import MeanSquaredError
from lib.neural_network.loss.binary_cross_entropy import BinaryCrossEntropy
from lib.neural_network.grid_search import GridSearch, Params
from lib.neural_network.loss.base import LossFunction
from lib.neural_network.activation.base import ActivationFunction

# ╭────────────────────────────────────────────────────────╮
# │             LOAD PREVIOUSLY PROCESSED DATA             │
# ╰────────────────────────────────────────────────────────╯
TARGET_COLUMN = "Diabetes_binary"

df_train = pd.read_csv("dataset/dataset_train.csv")
df_validation = pd.read_csv("dataset/dataset_validation.csv")

X_train = df_train.drop(columns=[TARGET_COLUMN]).astype(float)
y_train = df_train[TARGET_COLUMN].astype(int)

X_validation = df_validation.drop(columns=[TARGET_COLUMN]).astype(float)
y_validation = df_validation[TARGET_COLUMN].astype(int)

# ╭────────────────────────────────────────────────────────╮
# │               GRID SEARCH CONFIGURATION                │
# ╰────────────────────────────────────────────────────────╯

grid_search_params: Params = {
    "learning_rate": [
        0.1,
        # 0.01,
        # 0.001
    ],
    "learning_rate_patience": [3],
    "learning_rate_min": [1e-6],
    "learning_rate_max": [0.1],
    "batch_size": [
        500,
        1000
    ],
    "epochs": [500],
    "loss": [
        # MeanSquaredError(),
        BinaryCrossEntropy()
    ],
    "early_stopping_patience": [5],
    "architecture": [
        # Simple architecture : 1 hidden layer
        [
            {"neurons": [32], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [1], "dropout_rate": [0.0], "activation": [Sigmoid()]},
        ],
        # Medium architecture : 2 hidden layers
        [
            {"neurons": [64], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [32], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [1], "dropout_rate": [0.0], "activation": [Sigmoid()]},
        ],
        # Deep architecture : 3 hidden layers
        # [
        #     {"neurons": [128], "dropout_rate": [0.3], "activation": [Relu()]},
        #     {"neurons": [64], "dropout_rate": [0.2], "activation": [Relu()]},
        #     {"neurons": [32], "dropout_rate": [0.2], "activation": [Relu()]},
        #     {"neurons": [1], "dropout_rate": [0.0], "activation": [Sigmoid()]},
        # ],
        [
            {"neurons": [16], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [8], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [4], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [2], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [1], "dropout_rate": [0.0], "activation": [Sigmoid()]},
        ],
        [
            {"neurons": [32], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [8], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [4], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [2], "dropout_rate": [0.2], "activation": [Relu()]},
            {"neurons": [1], "dropout_rate": [0.0], "activation": [Sigmoid()]},
        ],
    ],
}

# ╭────────────────────────────────────────────────────────╮
# │                    RUN GRID SEARCH                     │
# ╰────────────────────────────────────────────────────────╯

gs = GridSearch(num_threads=1)

print("Starting grid search...")

date = time.strftime("%Y-%m-%d_%H-%M-%S")

results = gs.search_and_compare(
    grid_search_params,
    X_train.to_numpy(),
    y_train.to_numpy(),
    X_validation.to_numpy(),
    y_validation.to_numpy(),
    date=date,
    draw=False
)

# ╭────────────────────────────────────────────────────────╮
# │            SAVE GRID SEARCH RESULTS TO JSON            │
# ╰────────────────────────────────────────────────────────╯

json.dump(
    results,
    open(
        f"./grid_search_results/{date}.json",
        "w",
    ),
    indent=4,
    default=lambda o: (
        o.to_dict()
        if hasattr(o, "to_dict")
        else o if isinstance(o, (int, float, str, bool)) else str(o)
    ),
)
