from __future__ import annotations
import asyncio


class Callback:
    def on_epoch_begin(self, neural_network: "NeuralNetwork", epoch: int) -> None:  # type: ignore
        pass

    async def on_epoch_begin_async(
        self, neural_network: "NeuralNetwork", epoch: int  # type: ignore
    ) -> None:
        pass

    def on_epoch_end(self, neural_network: "NeuralNetwork", epoch: int) -> None:  # type: ignore
        pass

    async def on_epoch_end_async(
        self, neural_network: "NeuralNetwork", epoch: int  # type: ignore
    ) -> None:
        pass

    def on_batch_begin(self, neural_network: "NeuralNetwork", batch: int) -> None:  # type: ignore
        pass

    async def on_batch_begin_async(
        self, neural_network: "NeuralNetwork", batch: int  # type: ignore
    ) -> None:
        pass

    def on_batch_end(self, neural_network: "NeuralNetwork", batch: int) -> None:  # type: ignore
        pass

    async def on_batch_end_async(
        self, neural_network: "NeuralNetwork", batch: int  # type: ignore
    ) -> None:
        pass

    def on_train_begin(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        pass

    async def on_train_begin_async(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        pass

    def on_train_end(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        pass

    async def on_train_end_async(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        pass

    def on_train_cancel(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        pass

    async def on_train_cancel_async(self, neural_network: "NeuralNetwork") -> None:  # type: ignore
        pass
