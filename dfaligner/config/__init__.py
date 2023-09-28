from enum import Enum
from pathlib import Path
from typing import Optional, Union

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    BaseTrainingConfig,
    ConfigModel,
    PartialLoadConfig,
    init_context,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.config.utils import load_partials
from everyvoice.utils import load_config_from_json_or_yaml_path
from pydantic import Field, FilePath, ValidationInfo, model_validator


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


class DFAlignerConfig(PartialLoadConfig):
    # TODO FastSpeech2Config and DFAlignerConfig are almost identical.
    model: DFAlignerModelConfig = Field(default_factory=DFAlignerModelConfig)
    path_to_model_config_file: Optional[FilePath] = None

    training: DFAlignerTrainingConfig = Field(default_factory=DFAlignerTrainingConfig)
    path_to_training_config_file: Optional[FilePath] = None

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    path_to_preprocessing_config_file: Optional[FilePath] = None

    text: TextConfig = Field(default_factory=TextConfig)
    path_to_text_config_file: Optional[FilePath] = None

    @model_validator(mode="before")  # type: ignore
    def load_partials(self, info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,
            ("model", "training", "preprocessing", "text"),
            config_path=config_path,
        )

    @staticmethod
    def load_config_from_path(path: Path) -> "DFAlignerConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = DFAlignerConfig(**config)
        return config
