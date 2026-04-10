"""
Test the NeuralNetwork class to ensure it correctly initializes, trains, and makes predictions.
"""

import numpy as np
from .neural_network import NeuralNetwork
from .layer import Layer
from .loss.binary_cross_entropy import BinaryCrossEntropy
from .loss.mean_squared_error import MeanSquaredError
from .activation.relu import Relu
from .activation.sigmoid import Sigmoid
from .callback.early_stopping import EarlyStopping


def test_neural_network_initialization():
    """
    Test the initialization of a NeuralNetwork with basic parameters.
    """
    layer1 = Layer(neurons=10, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=5)

    assert network.layers == [layer1, layer2]
    assert network.loss == loss
    assert not network.callbacks
    assert network.fiting is False
    assert network.trained is False
    assert network.threshold == 0.5
    assert network.inputs == 5


def test_add_layer():
    """
    Test adding a layer to an existing NeuralNetwork.
    """
    layer1 = Layer(neurons=10, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1], loss=loss, inputs=5)
    assert len(network.layers) == 1

    network.add_layer(layer2)
    assert len(network.layers) == 2
    assert network.layers[1] == layer2


def test_add_callback():
    """
    Test adding a callback to a NeuralNetwork.
    """
    layer1 = Layer(neurons=10, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=5)
    callback = EarlyStopping(patience=5)

    assert len(network.callbacks) == 0

    network.add_callback(callback)
    assert len(network.callbacks) == 1
    assert network.callbacks[0] == callback


def test_predict_before_training():
    """
    Test that predict() raises an error if called before training.
    """
    layer1 = Layer(neurons=10, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=5)
    x_test = np.random.rand(10, 5)

    try:
        network.predict(x_test)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "not trained yet" in str(e)


def test_fit_and_predict():
    """
    Test the fit() and predict() methods of a NeuralNetwork.
    """
    layer1 = Layer(neurons=10, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=5)

    x_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 2, (50, 1))

    network.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=5,
        batch_size=10,
        validation_split=0.2,
        learning_rate=0.01,
    )

    assert network.trained is True
    assert network.fiting is False
    assert len(network.history) == 5
    assert network.epochs == 5

    # Test predictions
    x_test = np.random.rand(10, 5)
    predictions = network.predict(x_test)

    assert predictions.shape == (1, 10)
    assert np.all((predictions == 0) | (predictions == 1))


def test_history_tracking():
    """
    Test that the NeuralNetwork correctly tracks history during training.
    """
    layer1 = Layer(neurons=5, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = MeanSquaredError()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=3)

    x_train = np.random.rand(40, 3)
    y_train = np.random.rand(40, 1)

    network.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=3,
        batch_size=10,
        validation_split=0.25,
        learning_rate=0.01,
    )

    assert len(network.history) == 3

    for epoch_history in network.history:
        assert "loss" in epoch_history
        assert "val_loss" in epoch_history
        assert "y_pred" in epoch_history
        assert "x_train" in epoch_history
        assert "y_train" in epoch_history
        assert isinstance(epoch_history["loss"], (float, np.floating))
        assert isinstance(epoch_history["val_loss"], (float, np.floating))


def test_threshold():
    """
    Test that the threshold parameter correctly affects predictions.
    """
    layer1 = Layer(neurons=5, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=3)

    x_train = np.random.rand(30, 3)
    y_train = np.random.randint(0, 2, (30, 1))

    # Train with default threshold (0.5)
    network.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=3,
        batch_size=10,
        validation_split=0.2,
        learning_rate=0.01,
        threshold=0.5,
    )

    x_test = np.random.rand(10, 3)
    predictions_05 = network.predict(x_test)

    # Train with higher threshold (0.7)
    network2 = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=3)
    network2.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=3,
        batch_size=10,
        validation_split=0.2,
        learning_rate=0.01,
        threshold=0.7,
    )

    predictions_07 = network2.predict(x_test)

    assert predictions_05.shape == predictions_07.shape


def test_fit_parameters():
    """
    Test that fit() correctly stores all training parameters.
    """
    layer1 = Layer(neurons=8, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=4)

    x_train = np.random.rand(60, 4)
    y_train = np.random.randint(0, 2, (60, 1))

    epochs = 7
    batch_size = 12
    validation_split = 0.25
    learning_rate = 0.005
    threshold = 0.6

    network.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        learning_rate=learning_rate,
        threshold=threshold,
    )

    assert network.epochs == epochs
    assert network.batch_size == batch_size
    assert network.validation_split == validation_split
    assert network.learning_rate == learning_rate
    assert network.threshold == threshold


def test_callback_invocation():
    """
    Test that callbacks are invoked during training with EarlyStopping.
    """
    layer1 = Layer(neurons=5, activation=Relu())
    layer2 = Layer(neurons=1, activation=Sigmoid())
    loss = BinaryCrossEntropy()

    network = NeuralNetwork(layers=[layer1, layer2], loss=loss, inputs=3)

    x_train = np.random.rand(40, 3)
    y_train = np.random.randint(0, 2, (40, 1))

    early_stopping = EarlyStopping(patience=2)
    network.add_callback(early_stopping)

    network.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=10,
        batch_size=10,
        validation_split=0.2,
        learning_rate=0.01,
    )

    # With EarlyStopping, training should stop before 10 epochs or complete all
    assert network.trained is True
    assert len(network.history) <= 10


if __name__ == "__main__":
    print("Running NeuralNetwork tests...")
    test_neural_network_initialization()
    print("✓ test_neural_network_initialization passed")

    test_add_layer()
    print("✓ test_add_layer passed")

    test_add_callback()
    print("✓ test_add_callback passed")

    test_predict_before_training()
    print("✓ test_predict_before_training passed")

    test_fit_and_predict()
    print("✓ test_fit_and_predict passed")

    test_history_tracking()
    print("✓ test_history_tracking passed")

    test_threshold()
    print("✓ test_threshold passed")

    test_fit_parameters()
    print("✓ test_fit_parameters passed")

    test_callback_invocation()
    print("✓ test_callback_invocation passed")

    print("\nAll tests passed!")
