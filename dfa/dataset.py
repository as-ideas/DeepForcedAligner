from pathlib import Path
from typing import List, Dict, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
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


def collate_dataset(batch: List[dict]) -> torch.tensor:
    tokens = [b['tokens'] for b in batch]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    mels = [b['mel'] for b in batch]
    mels = pad_sequence(mels, batch_first=True, padding_value=0)
    tokens_len = torch.tensor([b['tokens_len'] for b in batch]).long()
    mel_len = torch.tensor([b['mel_len'] for b in batch]).long()
    return {'tokens': tokens, 'mels': mels, 'tokens_len': tokens_len, 'mel_len': mel_len}


def new_dataloader(dataset_path: Path, mel_dir: Path,
                   token_dir: Path, batch_size=32) -> DataLoader:
    dataset = unpickle_binary(dataset_path)
    item_ids = [d['item_id'] for d in dataset]
    aligner_dataset = AlignerDataset(item_ids=item_ids, mel_dir=mel_dir, token_dir=token_dir)
    return DataLoader(aligner_dataset,
                      collate_fn=collate_dataset,
                      batch_size=batch_size,
                      sampler=None,
                      num_workers=0,
                      pin_memory=True)


if __name__ == '__main__':
    config = read_config('config.yaml')
    paths = Paths(**config['paths'])
    dataloader = new_dataloader(paths.data_dir / 'dataset.pkl', paths.mel_dir, paths.token_dir)
    for i, b in enumerate(dataloader):
        print(f'{i}, {b}')
