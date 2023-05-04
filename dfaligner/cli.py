from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from everyvoice.base_cli.interfaces import (
    preprocess_base_command_interface,
    train_base_command_interface,
)
from loguru import logger
from merge_args import merge_args
from tqdm import tqdm

from .config import CONFIGS, DFAlignerConfig

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="A fork of the DeepForcedAligner project implemented in PyTorch Lightning",
)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    spec = "spec"
    text = "text"


@app.command()
@merge_args(preprocess_base_command_interface)
def preprocess(
    name: CONFIGS_ENUM = typer.Option(None, "--name", "-n"),
    data: Optional[List[PreprocessCategories]] = typer.Option(None, "-d", "--data"),
    **kwargs,
):
    from everyvoice.base_cli.helpers import preprocess_base_command

    preprocess_base_command(
        name=name,
        configs=CONFIGS,
        model_config=DFAlignerConfig,
        data=data,
        preprocess_categories=PreprocessCategories,
        **kwargs,
    )


@app.command()
@merge_args(train_base_command_interface)
def train(name: CONFIGS_ENUM = typer.Option(None, "--name", "-n"), **kwargs):
    from everyvoice.base_cli.helpers import train_base_command

    from .dataset import AlignerDataModule
    from .model import Aligner

    train_base_command(
        name=name,
        model_config=DFAlignerConfig,
        configs=CONFIGS,
        model=Aligner,
        data_module=AlignerDataModule,
        monitor="validation/loss",
        **kwargs,
    )


@app.command()
def extract_alignments(
    name: CONFIGS_ENUM = typer.Option(None, "--name", "-n"),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option("auto", "--devices", "-d"),
    model_path: Path = typer.Option(
        None, "--model", "-m", exists=True, file_okay=True, dir_okay=False
    ),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
    config_path: Path = typer.Option(
        None, "--config-path", "-p", exists=True, dir_okay=False, file_okay=True
    ),
    num_processes: int = typer.Option(None),
    predict: bool = typer.Option(True),
    create_n_textgrids: int = typer.Option(5, "--tg", "--n_textgrids"),
):
    from everyvoice.utils import update_config_from_cli_args

    from .utils import create_textgrid, extract_durations_for_item

    if config_path:
        config = DFAlignerConfig.load_config_from_path(config_path)
    elif name:
        config = DFAlignerConfig.load_config_from_path(CONFIGS[name.value])
    else:
        logger.error(
            "You must either choose a <NAME> of a preconfigured dataset, or provide a <CONFIG_PATH> to a preprocessing configuration file."
        )
        exit()

    config = update_config_from_cli_args(config_args, config)

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
    pbar.set_description("Loading EveryVoice modules")
    from .dataset import AlignerDataModule

    pbar.update()
    from everyvoice.text import TextProcessor

    from .model import Aligner

    pbar.update()

    # TODO: make this faster
    if num_processes is None:
        num_processes = 4

    data = AlignerDataModule(config)
    save_dir = Path(config.preprocessing.save_dir)
    (save_dir / "duration").mkdir(exist_ok=True, parents=True)
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
