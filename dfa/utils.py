import pickle
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

import numpy as np
import yaml

from .duration_extraction import (
    extract_durations_with_dijkstra,
    extract_durations_beam,
)


def read_metafile(path: str) -> Dict[str, str]:
    text_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            split = line.split("|")
            text_id, text = split[0], split[-1]
            text_dict[text_id] = text.strip()
    return text_dict


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, "w+", encoding="utf-8") as stream:
        yaml.dump(config, stream, default_flow_style=False)


def get_files(path: str, extension=".wav") -> List[Path]:
    return list(Path(path).expanduser().resolve().rglob(f"*{extension}"))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    with open(str(file), "wb") as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> Any:
    with open(str(file), "rb") as f:
        return pickle.load(f)


def extract_durations_for_item(item, tokens, pred, method: str = "beam"):
    tokens_len, mel_len = item["tokens_len"], item["mel_len"]
    tokens = tokens[:tokens_len]
    pred = pred[:mel_len, :]
    if method == "beam":
        durations, _ = extract_durations_beam(tokens, pred, 10)
        durations = durations[0]
    elif method == "dijkstra":
        durations = extract_durations_with_dijkstra(tokens, pred)
    else:
        raise NotImplementedError(f"Sorry, method '{method}' is not implemented")

    return item, durations
