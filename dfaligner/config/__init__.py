from enum import Enum
from pathlib import Path
from typing import Union

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    BaseTrainingConfig,
    ConfigModel,
    PartialConfigModel,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.utils import load_config_from_json_or_yaml_path
from pydantic import Field


class DFAlignerExtractionMethod(Enum):
    beam = "beam"
    dijkstra = "dijkstra"


class DFAlignerModelConfig(ConfigModel):
    lstm_dim: int = 512
    conv_dim: int = 512


class DFAlignerTrainingConfig(BaseTrainingConfig):
    optimizer: Union[AdamOptimizer, AdamWOptimizer] = Field(
        default_factory=AdamWOptimizer
    )
    binned_sampler: bool = True
    plot_steps: int = 1000
    extraction_method: DFAlignerExtractionMethod = DFAlignerExtractionMethod.dijkstra


class DFAlignerConfig(PartialConfigModel):
    model: DFAlignerModelConfig = Field(default_factory=DFAlignerModelConfig)
    training: DFAlignerTrainingConfig = Field(default_factory=DFAlignerTrainingConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    text: TextConfig = Field(default_factory=TextConfig)

    @staticmethod
    def load_config_from_path(path: Path) -> "DFAlignerConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return DFAlignerConfig(**config)