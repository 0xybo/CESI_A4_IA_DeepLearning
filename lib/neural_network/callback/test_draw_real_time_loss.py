import asyncio
import numpy as np
import matplotlib.pyplot as plt
from .draw_real_time_loss import DrawRealTimeLoss
from ..neural_network import NeuralNetwork
from ..loss.binary_cross_entropy import BinaryCrossEntropy


class NeuralNetworkPlaceholder(NeuralNetwork):
    def __init__(self):
        super().__init__(layers=[], loss=BinaryCrossEntropy())
        self.history = [
            {
                "loss": 0.5,
                "val_loss": 0.6,
                "y_pred": np.array([[0.5], [0.5]]),
                "x_train": np.array([[0], [1]]),
                "y_train": np.array([[0], [1]]),
            }
        ]


async def test_draw_real_time_loss():
    neural_network_placeholder = NeuralNetworkPlaceholder()
    callback = DrawRealTimeLoss()

    assert callback.figure is not None
    assert callback.axes is not None
    assert callback.loss_line is not None
    assert callback.val_loss_line is not None

    for epoch in range(100):
        neural_network_placeholder.history.append(
            {
                "loss": 0.5 / (epoch + 1),
                "val_loss": 0.6 / (epoch + 1),
                "y_pred": neural_network_placeholder.predicts(
                    neural_network_placeholder.history[0]["x_train"]
                ),
                "x_train": neural_network_placeholder.history[0]["x_train"],
                "y_train": neural_network_placeholder.history[0]["y_train"],
            }
        )

        await callback.on_epoch_end_async(neural_network_placeholder, epoch)


if __name__ == "__main__":
    asyncio.run(test_draw_real_time_loss())
    plt.show()
