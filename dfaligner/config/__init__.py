from enum import Enum
from pathlib import Path
from typing import Optional, Union

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    BaseTrainingConfig,
    ConfigModel,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.config.utils import load_partials
from everyvoice.utils import load_config_from_json_or_yaml_path
from pydantic import Field, FilePath, model_validator


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


class DFAlignerConfig(ConfigModel):
    model: DFAlignerModelConfig = Field(default_factory=DFAlignerModelConfig)
    path_to_model_config_file: Optional[FilePath] = None

    training: DFAlignerTrainingConfig = Field(default_factory=DFAlignerTrainingConfig)
    path_to_training_config_file: Optional[FilePath] = None

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    path_to_preprocessing_config_file: Optional[FilePath] = None

    text: TextConfig = Field(default_factory=TextConfig)
    path_to_text_config_file: Optional[FilePath] = None

    @model_validator(mode="before")
    def load_partials(self):
        return load_partials(self, ["model", "training", "preprocessing", "text"])

    @staticmethod
    def load_config_from_path(path: Path) -> "DFAlignerConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return DFAlignerConfig(**config)
