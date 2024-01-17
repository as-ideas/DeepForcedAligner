import pickle
from pathlib import Path
from typing import Any

import yaml
from pympi.Praat import TextGrid

from .duration_extraction import extract_durations_beam, extract_durations_with_dijkstra


def create_textgrid(save_path, tokens, durations, hop_size, sample_rate):
    tg = TextGrid(xmax=(sum(durations) * hop_size) / sample_rate)
    token_tier = tg.add_tier(name="Tokens")
    current_time = 0
    for i, d in enumerate(durations):
        dur_seconds = (d * hop_size) / sample_rate
        token_tier.add_interval(
            begin=current_time, end=current_time + dur_seconds, value=tokens[i]
        )
        current_time += dur_seconds
    tg.to_file(save_path)


def read_metafile(path: str) -> dict[str, str]:
    text_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            split = line.split("|")
            text_id, text = split[0], split[-1]
            text_dict[text_id] = text.strip()
    return text_dict


def read_config(path: str) -> dict[str, Any]:
    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: dict[str, Any], path: str) -> None:
    with open(path, "w+", encoding="utf-8") as stream:
        yaml.dump(config, stream, default_flow_style=False)


def get_files(path: str, extension=".wav") -> list[Path]:
    return list(Path(path).expanduser().resolve().rglob(f"*{extension}"))


def pickle_binary(data: object, file: str | Path) -> None:
    with open(str(file), "wb") as f:
        pickle.dump(data, f)


def unpickle_binary(file: str | Path) -> Any:
    with open(str(file), "rb") as f:
        return pickle.load(f)


def extract_durations_for_item(item, tokens, pred, method: str = "beam"):
    tokens_len, mel_len = item["tokens_len"], item["mel_len"]
    tokens = tokens[:tokens_len]
    pred = pred[:mel_len, :]
    if method == "beam":
        duration_candidates, _ = extract_durations_beam(tokens, pred, 10)
        durations = duration_candidates[0]
    elif method == "dijkstra":
        durations = extract_durations_with_dijkstra(tokens, pred)
    else:
        raise NotImplementedError(f"Sorry, method '{method}' is not implemented")

    return item, durations
