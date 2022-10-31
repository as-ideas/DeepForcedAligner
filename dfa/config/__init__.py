from enum import Enum
from pathlib import Path
from typing import Dict, Union
import yaml

from smts.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    ConfigModel,
    PartialConfigModel,
    BaseTrainingConfig,
)
from smts.config.text_config import TextConfig
from smts.config.preprocessing_config import PreprocessingConfig


class DFAlignerExtractionMethod(Enum):
    beam = "beam"
    dijkstra = "dijkstra"


class DFAlignerModelConfig(ConfigModel):
    lstm_dim: int
    conv_dim: int


class DFAlignerTrainingConfig(BaseTrainingConfig):
    optimizer: Union[AdamOptimizer, AdamWOptimizer]
    binned_sampler: bool
    plot_steps: int
    extraction_method: DFAlignerExtractionMethod


class DFAlignerConfig(PartialConfigModel):
    model: DFAlignerModelConfig
    training: DFAlignerTrainingConfig
    preprocessing: PreprocessingConfig
    text: TextConfig

    @staticmethod
    def load_config_from_path(path: Path) -> dict:
        """Load a config from a path"""
        with open(path) as f:
            config = yaml.safe_load(f)
        return DFAlignerConfig(**config)


CONFIGS: Dict[str, DFAlignerConfig] = {
    "base": DFAlignerConfig.load_config_from_path(Path(__file__).parent / "base.yaml"),
}
