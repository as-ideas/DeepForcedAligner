from pathlib import Path
from random import Random
from typing import List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from dfa.utils import unpickle_binary


class AlignerDataset(Dataset):

    def __init__(self, item_ids: List[str], mel_dir: Path, token_dir: Path):
        super().__init__()
        self.item_ids = item_ids
        self.mel_dir = mel_dir
        self.token_dir = token_dir

    def __getitem__(self, index):
        item_id = self.item_ids[index]
        mel = np.load(str(self.mel_dir / f'{item_id}.npy'), allow_pickle=False)
        tokens = np.load(str(self.token_dir / f'{item_id}.npy'), allow_pickle=False)
        mel = torch.tensor(mel).float()
        tokens = torch.tensor(tokens).long()

        return {'item_id': item_id, 'tokens': tokens, 'mel': mel,
                'tokens_len': tokens.size(0), 'mel_len': mel.size(0)}

    def __len__(self):
        return len(self.item_ids)


# From https://github.com/fatchord/WaveRNN/blob/master/utils/dataset.py
class BinnedLengthSampler(Sampler):

    def __init__(self, mel_lens: torch.tensor, batch_size: int, bin_size: int, seed=42):
        _, self.idx = torch.sort(torch.tensor(mel_lens))
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.random = Random(seed)
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []
        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            self.random.shuffle(this_bin)
            bins += [this_bin]
        self.random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)
        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            self.random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])
        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


def collate_dataset(batch: List[dict]) -> torch.tensor:
    tokens = [b['tokens'] for b in batch]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    mels = [b['mel'] for b in batch]
    mels = pad_sequence(mels, batch_first=True, padding_value=0)
    tokens_len = torch.tensor([b['tokens_len'] for b in batch]).long()
    mel_len = torch.tensor([b['mel_len'] for b in batch]).long()
    item_ids = [b['item_id'] for b in batch]
    return {'tokens': tokens, 'mel': mels, 'tokens_len': tokens_len,
            'mel_len': mel_len, 'item_id': item_ids}


def new_dataloader(dataset_path: Path, mel_dir: Path,
                   token_dir: Path, batch_size=32) -> DataLoader:
    dataset = unpickle_binary(dataset_path)
    print(f'len data {len(dataset)}')
    dataset = [d for d in dataset if d['mel_len'] < 1250]
    item_ids = [d['item_id'] for d in dataset]
    mel_lens = [d['mel_len'] for d in dataset]
    print(f'len filtered data {len(dataset)}')


    aligner_dataset = AlignerDataset(item_ids=item_ids, mel_dir=mel_dir, token_dir=token_dir)
    return DataLoader(aligner_dataset,
                      collate_fn=collate_dataset,
                      batch_size=batch_size,
                      sampler=BinnedLengthSampler(mel_lens=mel_lens, batch_size=batch_size,
                                                  bin_size=batch_size*3),
                      num_workers=0,
                      pin_memory=True)


def get_longest_mel_id(dataset_path: Path) -> str:
    dataset = unpickle_binary(dataset_path)
    dataset = [d for d in dataset if d['mel_len'] < 1250]
    dataset.sort(key=lambda item: (item['mel_len'], item['item_id']))
    return dataset[-1]['item_id']
