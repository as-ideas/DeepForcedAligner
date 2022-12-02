import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from smts.text import TextProcessor
from tqdm import tqdm

from .config import CONFIGS, DFAlignerConfig
from .utils import create_textgrid, extract_durations_for_item

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
    from smts.preprocessor import Preprocessor

    config = DFAlignerConfig.load_config_from_path(CONFIGS[name.value])
    preprocessor = Preprocessor(config)
    to_preprocess = {k: k in data for k in PreprocessCategories.__members__.keys()}  # type: ignore
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (pitch, mel, energy, durations, inputs) from dataset '{name}'"
        )
        to_preprocess = {k: True for k in to_preprocess}
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
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        "-a",
        help="Uses PyTorch Lightning Accelerators: https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html",
    ),
    devices: str = typer.Option("auto", "--devices", "-d"),
    strategy: str = typer.Option(None),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
):
    logger.info("Loading modules for alignment...")
    pbar = tqdm(range(6))
    pbar.set_description("Loading pytorch and friends")
    from pytorch_lightning import Trainer

    pbar.update()
    pbar.refresh()
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    pbar.update()
    pbar.refresh()
    from pytorch_lightning.loggers import TensorBoardLogger

    pbar.update()
    pbar.refresh()
    pbar.set_description("Loading SmallTeamSpeech modules")
    from smts.utils import update_config_from_cli_args, update_config_from_path

    pbar.update()
    pbar.refresh()

    from .dataset import AlignerDataModule

    pbar.update()
    pbar.refresh()
    from .model import Aligner

    pbar.update()
    pbar.refresh()

    original_config = DFAlignerConfig.load_config_from_path(CONFIGS[name.value])
    config: DFAlignerConfig = update_config_from_cli_args(config_args, original_config)
    config = update_config_from_path(config_path, config)
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
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option("auto", "--devices", "-d"),
    model_path: Path = typer.Option(
        None, "--model", "-m", exists=True, file_okay=True, dir_okay=False
    ),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
    num_processes: int = typer.Option(None),
    predict: bool = typer.Option(True),
    create_n_textgrids: int = typer.Option(5, "--tg", "--n_textgrids"),
):
    from smts.utils import update_config_from_cli_args, update_config_from_path

    original_config = DFAlignerConfig.load_config_from_path(CONFIGS[name.value])
    config: DFAlignerConfig = update_config_from_cli_args(config_args, original_config)
    config = update_config_from_path(config_path, config)

    # Imports
    logger.info("Loading modules for alignment...")
    pbar = tqdm(range(6))
    pbar.set_description("Loading pytorch and friends")
    import numpy as np

    pbar.update()
    import torch

    pbar.update()
    from pytorch_lightning.loggers import TensorBoardLogger

    pbar.update()
    from pytorch_lightning import Trainer

    pbar.update()
    pbar.set_description("Loading SmallTeamSpeech modules")
    from .dataset import AlignerDataModule

    pbar.update()
    from .model import Aligner

    pbar.update()

    # TODO: make this faster
    if num_processes is None:
        num_processes = 4

    data = AlignerDataModule(config)
    if predict:
        tensorboard_logger = TensorBoardLogger(**(config.training.logger.dict()))
        trainer = Trainer(
            accelerator=accelerator, devices=devices, logger=tensorboard_logger
        )
        if model_path:
            model: Aligner = Aligner.load_from_checkpoint(
                model_path.absolute().as_posix()
            )
            # TODO: check into the best way to update config from re-loaded model
            # model.update_config(config)
            model.config = config
            trainer.predict(model, dataloaders=data)
        else:
            trainer.predict(dataloaders=data)
    sep = config.preprocessing.value_separator
    save_dir = Path(config.preprocessing.save_dir)
    tg_processed = 0
    text_processor = TextProcessor(config)
    for item in tqdm(
        data.predict_dataloader().dataset,
        total=len(data.predict_dataloader().dataset),
    ):
        basename = item["basename"]
        speaker = item["speaker"]
        language = item["language"]
        tokens = item["tokens"].cpu()
        pred = np.load(
            save_dir
            / "duration"
            / sep.join([basename, speaker, language, "duration.npy"])
        )
        item, durations = extract_durations_for_item(
            item,
            tokens,
            pred,
            # ignore mypy type checking because https://github.com/pydantic/pydantic/issues/3809
            method=config.training.extraction_method,  # type: ignore
        )
        assert len(durations) == len(
            tokens
        ), f"Length of tokens and durations must be the same, but was not for {basename}. Try re-running with dijkstra extraction method. This might be because your model was not trained properly or because you are showing it unseen data."
        if tg_processed < create_n_textgrids:
            (save_dir / "text_grid").mkdir(parents=True, exist_ok=True)
            create_textgrid(
                save_dir
                / "text_grid"
                / sep.join([basename, speaker, language, "duration.TextGrid"]),
                text_processor.token_sequence_to_text_sequence(tokens.tolist()),
                durations,
                hop_size=config.preprocessing.audio.fft_hop_frames,
                sample_rate=config.preprocessing.audio.alignment_sampling_rate,
            )
            tg_processed += 1
        torch.save(
            torch.tensor(durations).long(),
            save_dir
            / "duration"
            / sep.join([basename, speaker, language, "duration.pt"]),
        )


if __name__ == "__main__":
    app()
