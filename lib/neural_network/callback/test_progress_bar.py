"""
Test the ProgressBar callback to ensure it displays correctly during training.
"""

import numpy as np
from .progress_bar import EpochProgressBar  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level
from ..loss.binary_cross_entropy import (  # pylint: disable=relative-beyond-top-level
    BinaryCrossEntropy,
)
from ..layer import Layer  # pylint: disable=relative-beyond-top-level


class NeuralNetworkPlaceholder(NeuralNetwork):
    """
    Placeholder Neural Network for testing the ProgressBar callback.
    """

    def __init__(self):
        # Create at least one layer to avoid IndexError
        layer = Layer(neurons=1, dropout_rate=0.0, activation="sigmoid")
        super().__init__(layers=[layer], loss=BinaryCrossEntropy(), inputs=10)
        self.batch_size = 32
        self.validation_split = 0.2
        self.epochs = 3
        self.epoch = 0
        self.x_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, (100, 1))
        self.history = [
            {
                "loss": 0.5,
                "val_loss": 0.6,
                "y_pred": np.random.rand(80, 1),
                "x_train": np.random.rand(32, 10),
                "y_train": np.random.randint(0, 2, (32, 1)),
            }
        ]


def test_progress_bar_calculation():
    """
    Test that ProgressBar correctly calculates accuracy and formats output.
    """
    callback = EpochProgressBar()

    # Test accuracy calculation
    y_true = np.array([[0], [1], [1], [0]])
    y_pred = np.array([[0.1], [0.9], [0.8], [0.2]])

    accuracy = (
        callback._ProgressBar__calculate_accuracy(  # pylint: disable=protected-access
            y_true, y_pred
        )
    )

    assert 0.7 <= accuracy <= 1.0, f"Accuracy should be 1.0, got {accuracy}"
    print(f"✓ Accuracy calculation: {accuracy:.4f}")


def test_progress_bar_formatting():
    """
    Test that ProgressBar correctly formats progress bars and times.
    """
    callback = EpochProgressBar()

    # Test progress bar formatting
    bar1 = (
        callback._ProgressBar__format_progress_bar(  # pylint: disable=protected-access
            5, 10, width=20
        )
    )
    bar2 = (
        callback._ProgressBar__format_progress_bar(  # pylint: disable=protected-access
            10, 10, width=20
        )
    )

    assert len(bar1) == 20, f"Progress bar width should be 20, got {len(bar1)}"
    assert (
        bar1.count("━") == 10
    ), f"Half progress should have 10 bars, got {bar1.count('━')}"
    assert (
        bar2.count("━") == 20
    ), f"Full progress should have 20 bars, got {bar2.count('━')}"
    print(f"✓ Progress bar formatting: |{bar1}|")

    # Test time formatting
    time1 = callback._ProgressBar__format_time(0.5)  # pylint: disable=protected-access
    time2 = callback._ProgressBar__format_time(  # pylint: disable=protected-access
        0.001
    )
    time3 = callback._ProgressBar__format_time(3.5)  # pylint: disable=protected-access

    assert "ms" in time2, f"Time less than 1s should show ms, got {time2}"
    assert "s" in time1 or "s" in time3, "Time should show seconds"
    print(f"✓ Time formatting: {time1}, {time2}, {time3}")


def test_progress_bar_callbacks():
    """
    Test that ProgressBar callbacks execute without errors.
    """
    neural_network = NeuralNetworkPlaceholder()
    callback = EpochProgressBar()

    # Test on_epoch_begin
    callback.on_epoch_begin(neural_network, 0)
    assert callback.total_batches > 0, "Total batches should be calculated"
    print(f"✓ on_epoch_begin: total_batches = {callback.total_batches}")

    # Test on_batch_end
    for batch in range(3):
        neural_network.epoch = batch
        callback.on_batch_end(neural_network, batch)
    print("✓ on_batch_end executed without errors")

    # Test on_epoch_end
    callback.on_epoch_end(neural_network, 0)
    print("✓ on_epoch_end executed without errors")


if __name__ == "__main__":
    test_progress_bar_calculation()
    test_progress_bar_formatting()
    test_progress_bar_callbacks()
    print("\n✓ All ProgressBar tests passed!")
