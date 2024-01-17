from enum import Enum
from pathlib import Path
from typing import Any, Optional

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
    lstm_dim: int = Field(
        512, description="The number of dimensions in the LSTM layers."
    )
    conv_dim: int = Field(
        512, description="The number of dimensions in the convolutional layers."
    )


class DFAlignerTrainingConfig(BaseTrainingConfig):
    optimizer: AdamOptimizer | AdamWOptimizer = Field(
        default_factory=AdamWOptimizer,
        description="Optimizer configuration settings.",
    )
    binned_sampler: bool = Field(True, description="Use a binned length sampler")
    plot_steps: int = Field(1000, description="The maximum number of steps to plot")
    extraction_method: DFAlignerExtractionMethod = Field(
        DFAlignerExtractionMethod.dijkstra,
        description="The alignment extraction algorithm to use. 'beam' will be quicker but possibly less accurate than 'dijkstra'",
    )


class DFAlignerConfig(PartialLoadConfig):
    # TODO FastSpeech2Config and DFAlignerConfig are almost identical.
    model: DFAlignerModelConfig = Field(
        default_factory=DFAlignerModelConfig,
        description="The model configuration settings.",
    )
    path_to_model_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    training: DFAlignerTrainingConfig = Field(
        default_factory=DFAlignerTrainingConfig,
        description="The training configuration hyperparameters.",
    )
    path_to_training_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="The preprocessing configuration, including information about audio settings.",
    )
    path_to_preprocessing_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    text: TextConfig = Field(default_factory=TextConfig)
    path_to_text_config_file: Optional[FilePath] = None

    @model_validator(mode="before")  # type: ignore
    def load_partials(self: dict[Any, Any], info: ValidationInfo):
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
