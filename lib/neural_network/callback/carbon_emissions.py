from .base import Callback  # pylint: disable=relative-beyond-top-level
from ..neural_network import NeuralNetwork  # pylint: disable=relative-beyond-top-level
from codecarbon import EmissionsTracker
from codecarbon.output import EmissionsData
from pandas import DataFrame

class CarbonEmissions(Callback):


    tracker: EmissionsTracker
    data: EmissionsData

    def __init__(self) -> None:
        super().__init__()
        self.tracker = EmissionsTracker(
            save_to_file=False,
            tracking_mode='process',
            log_level="error"
        )

    def on_train_begin(self, _neural_network: "NeuralNetwork") -> None:  # type: ignore
        self.tracker.start()
        
    def on_train_end(self, _neural_network: "NeuralNetwork") -> None:  # type: ignore
        self.tracker.stop()
        self.data = self.tracker.final_emissions_data
        
    def draw_result(self) -> DataFrame:
        return DataFrame({
            "timestamp": [self.data.timestamp],
            "duration": [self.data.duration],
            "emissions": [self.data.emissions],
            "emissions_rate": [self.data.emissions_rate],
            "energy_consumed": [self.data.energy_consumed],
        })
