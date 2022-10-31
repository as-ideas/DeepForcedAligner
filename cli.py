import json
import os
import typer
import yaml

import numpy as np
import torch
from enum import Enum
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List, Optional
from tqdm import tqdm
from loguru import logger
from smts.preprocessor import Preprocessor
from smts.utils import expand_config_string_syntax

from dfa.config import CONFIGS, DFAlignerConfig
from dfa.dataset import AlignerDataModule
from dfa.model import Aligner
from dfa.utils import extract_durations_for_item


app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    mel = "mel"
    text = "text"


@app.command()
def preprocess(
    name: CONFIGS_ENUM,
    data: Optional[List[PreprocessCategories]] = typer.Option(None, "-d", "--data"),
    output_path: Optional[Path] = typer.Option(
        "processed_filelist.psv", "-o", "--output"
    ),
    overwrite: bool = typer.Option(False, "-O", "--overwrite"),
):
    config = CONFIGS[name.value]
    preprocessor = Preprocessor(config)
    to_preprocess = {k: k in data for k in PreprocessCategories.__members__.keys()}  # type: ignore
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (pitch, mel, energy, durations, inputs) from dataset '{name}'"
        )
    else:
        preprocessor.preprocess(
            output_path=output_path,
            process_audio=to_preprocess["audio"],
            process_spec=to_preprocess["mel"],
            process_text=to_preprocess["text"],
            overwrite=overwrite,
        )


@app.command()
def train(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto"),
    devices: str = typer.Option("auto"),
    strategy: str = typer.Option(None),
    config: List[str] = typer.Option(None),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
):
    original_config = CONFIGS[name.value]
    if config is not None and config:
        for update in config:
            key, value = update.split("=")
            logger.info(f"Updating config '{key}' to value '{value}'")
            original_config = original_config.update_config(
                original_config, expand_config_string_syntax(update)
            )
    else:
        config: DFAlignerConfig = original_config
    if config_path is not None:
        logger.info(f"Loading and updating config from '{config_path}'")
        config_override = json.load(config_path)
        config = config.update_config(config, config_override)
    tensorboard_logger = TensorBoardLogger(**(config.training.logger.dict()))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger.info("Starting training for alignment model.")
    ckpt_callback = ModelCheckpoint(
        monitor="validation/loss",
        mode="min",
        save_last=True,
        save_top_k=config.training.save_top_k_ckpts,
        every_n_train_steps=config.training.ckpt_steps,
        every_n_epochs=config.training.ckpt_epochs,
    )
    trainer = Trainer(
        gradient_clip_val=1.0,
        logger=tensorboard_logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.training.max_epochs,
        callbacks=[ckpt_callback, lr_monitor],
        strategy=strategy,
        detect_anomaly=False,  # used for debugging, but triples training time
    )
    aligner = Aligner(config)
    data = AlignerDataModule(config)
    last_ckpt = (
        config.training.finetune_checkpoint
        if config.training.finetune_checkpoint is not None
        and os.path.exists(config.training.finetune_checkpoint)
        else None
    )
    tensorboard_logger.log_hyperparams(config.dict())
    trainer.fit(aligner, data, ckpt_path=last_ckpt)


@app.command()
def extract_alignments(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto"),
    devices: str = typer.Option("auto"),
    model_path: Path = typer.Option(
        default=None, exists=True, file_okay=True, dir_okay=False
    ),
    config: List[str] = typer.Option(None),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
    num_processes: int = typer.Option(None),
):
    # TODO: make this faster
    if num_processes is None:
        num_processes = 4
    original_config = CONFIGS[name.value]
    if config is not None and config:
        for update in config:
            key, value = update.split("=")
            logger.info(f"Updating config '{key}' to value '{value}'")
            original_config = original_config.update_config(
                original_config, expand_config_string_syntax(update)
            )
    else:
        config = original_config
    if config_path is not None:
        logger.info(f"Loading and updating config from '{config_path}'")
        if config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config_override = json.load(f)
        if config_path.suffix == ".yaml":
            with open(config_path, "r") as f:
                config_override = yaml.safe_load(f)
        config: DFAlignerConfig = config.update_config(config, config_override)
    data = AlignerDataModule(config)
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
    )
    if model_path:
        model = Aligner.load_from_checkpoint(model_path)
        # TODO: check into the best way to update config from re-loaded model
        # model.update_config(config)
        model.config = config
        trainer.predict(model, dataloaders=data)
    else:
        trainer.predict(dataloaders=data)
    sep = config.preprocessing.value_separator
    save_dir = Path(config.preprocessing.save_dir)
    for item in tqdm(
        data.predict_dataloader().dataset,
        total=len(data.predict_dataloader().dataset),
    ):
        basename = item["basename"]
        speaker = item["speaker"]
        language = item["language"]
        tokens = item["tokens"].cpu()
        pred = np.load(
            save_dir / sep.join([basename, speaker, language, "duration.npy"])
        )
        item, durations = extract_durations_for_item(
            item,
            tokens,
            pred,
            method=config.training.extraction_method,
        )
        torch.save(
            torch.tensor(durations).long(),
            save_dir / sep.join([basename, speaker, language, "duration.npy"]),
        )


if __name__ == "__main__":
    app()
