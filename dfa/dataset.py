from pathlib import Path
from typing import List

import torch
from torch.utils.data.dataset import Dataset

from dfa.paths import Paths
from dfa.utils import unpickle_binary, read_config


class AlignerDataset(Dataset):

    def __init__(self, item_ids: List[str], mel_dir: Path, token_dir: Path):
        self.item_ids = item_ids
        self.mel_dir = mel_dir
        self.token_dir = token_dir

    def __getitem__(self, index):
        item_id = self.item_ids[index]
        mel = torch.load(self.mel_dir / f'{item_id}.pt')
        tokens = torch.load(self.token_dir / f'{item_id}.pt')
        return {'item_id': item_id, 'tokens': tokens, 'mel': mel,
                'tokens_len': tokens.size(0), 'mel_len': mel.size(1)}

    def __len__(self):
        return len(self.item_ids)


def new_aligner_dataset(dataset_path: Path, mel_dir: Path, token_dir: Path) -> AlignerDataset:
    dataset = unpickle_binary(dataset_path)
    item_ids = [d['item_id'] for d in dataset]
    return AlignerDataset(item_ids=item_ids, mel_dir=mel_dir, token_dir=token_dir)


if __name__ == '__main__':
    config = read_config('config.yaml')
    paths = Paths(**config['paths'])
    aligner_dataset = new_aligner_dataset(paths.data_dir / 'dataset.pkl', paths.mel_dir, paths.token_dir)

    for i, b in enumerate(aligner_dataset):
        print(f'{i}, {b.keys()}')
