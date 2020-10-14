import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Any, Union


def read_metafile(path: str) -> Dict[str, str]:
    csv_files = get_files(path, extension='.csv')
    assert len(csv_files) == 1, f'Found multiple csv files in {path}, expected a single one!'
    csv_file = str(csv_files[0])
    text_dict = {}
    with open(csv_file, encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            text_id, text = split[0], split[-1]
            text_dict[text_id] = text.strip()
    return text_dict


def read_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def get_files(path: str, extension='.wav') -> List[Path]:
    return list(Path(path).expanduser().resolve().rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> Any:
    with open(str(file), 'rb') as f:
        return pickle.load(f)
