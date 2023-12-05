import os
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from everyvoice.text import TextProcessor
from everyvoice.utils import check_dataset_size
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from .config import DFAlignerConfig


class AlignerDataModule(pl.LightningDataModule):
    def __init__(self, config: DFAlignerConfig):
        super().__init__()
        self.config = config
        self.train_sampler = None
        self.val_sampler = None
        self.batch_size = config.training.batch_size
        self.train_path = os.path.join(
            config.training.logger.save_dir,
            config.training.logger.name,
            "alignment_train_data.pth",
        )
        self.val_path = os.path.join(
            config.training.logger.save_dir,
            config.training.logger.name,
            "alignment_val_data.pth",
        )

        self.load_dataset()
        self.dataset_length = len(self.train_dataset) + len(self.val_dataset)

    def setup(self, stage: Optional[str] = None):
        # load it back here
        self.train_dataset = torch.load(self.train_path)
        self.val_dataset = torch.load(self.val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=False,
            sampler=self.train_sampler,
            collate_fn=collate_dataset,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            pin_memory=False,
            sampler=self.val_sampler,
            collate_fn=collate_dataset,
            drop_last=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            AlignerDataset(self.train_dataset + self.val_dataset, self.config),
            batch_size=self.batch_size,
            pin_memory=False,
            collate_fn=collate_dataset,
            drop_last=True,
        )

    def prepare_data(self):
        train_samples = len(self.train_dataset)
        val_samples = len(self.val_dataset)
        check_dataset_size(self.batch_size, train_samples, "training")
        check_dataset_size(self.batch_size, val_samples, "validation")
        self.train_dataset = AlignerDataset(self.train_dataset, self.config)
        self.val_dataset = AlignerDataset(self.val_dataset, self.config)
        if self.config.training.binned_sampler:
            self.train_mel_lens = [d["mel_len"] for d in self.train_dataset]
            self.val_mel_lens = [d["mel_len"] for d in self.val_dataset]
            self.train_sampler = BinnedLengthSampler(
                mel_lens=self.train_mel_lens,
                batch_size=self.batch_size,
                bin_size=self.batch_size * 3,
                seed=self.config.training.seed,
            )
            self.val_sampler = BinnedLengthSampler(
                mel_lens=self.val_mel_lens,
                batch_size=self.batch_size,
                bin_size=self.batch_size * 3,
                seed=self.config.training.seed,
            )

        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)

    def load_dataset(self):
        # Can use same filelist as for feature prediction
        self.train_dataset = self.config.training.filelist_loader(
            self.config.training.training_filelist
        )
        self.val_dataset = self.config.training.filelist_loader(
            self.config.training.validation_filelist
        )


class AlignerDataset(Dataset):
    def __init__(self, data, config: DFAlignerConfig):
        super().__init__()
        self.config = config
        self.data = data
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        self.text_processor = TextProcessor(config)
        self.sep = "--"
        self.sampling_rate = self.config.preprocessing.audio.alignment_sampling_rate

    def _load_file(self, bn, spk, lang, dir, fn):
        return torch.load(
            self.preprocessed_dir / dir / self.sep.join([bn, spk, lang, fn])
        )

    def __getitem__(self, index):
        item = self.data[index]
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        basename = item["basename"]
        mel = (
            self._load_file(
                basename,
                speaker,
                language,
                "spec",
                f"spec-{self.sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
            )
            .squeeze()
            .transpose(0, 1)
        )  # [mel_bins, frames] -> [frames, mel_bins]
        text_tokens = self._load_file(basename, speaker, language, "text", "text.pt")
        tokens_len = text_tokens.size(0)
        mel_len = mel.size(0)
        return {
            "basename": basename,
            "tokens": text_tokens,
            "mel": mel,
            "tokens_len": tokens_len,
            "mel_len": mel_len,
            "speaker": speaker,
            "language": language,
        }

    def __len__(self):
        return len(self.data)


# From https://github.com/fatchord/WaveRNN/blob/master/utils/dataset.py
class BinnedLengthSampler(Sampler):
    def __init__(self, mel_lens: torch.Tensor, batch_size: int, bin_size: int, seed=42):
        _, self.idx = torch.sort(torch.tensor(mel_lens))
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.random = Random(seed)
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []
        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size : (i + 1) * self.bin_size]
            self.random.shuffle(this_bin)
            bins += [this_bin]
        self.random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)
        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx) :]
            self.random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])
        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


def collate_dataset(batch: List[dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    tokens: torch.Tensor = pad_sequence(
        [b["tokens"] for b in batch], batch_first=True, padding_value=0
    )
    mels: torch.Tensor = pad_sequence(
        [b["mel"] for b in batch], batch_first=True, padding_value=0
    )
    tokens_len = torch.tensor([b["tokens_len"] for b in batch]).long()
    mel_len = torch.tensor([b["mel_len"] for b in batch]).long()
    return {
        "tokens": tokens,
        "mel": mels,
        "tokens_len": tokens_len,
        "mel_len": mel_len,
        "basename": [b["basename"] for b in batch],
        "language": [b["language"] for b in batch],
        "speaker": [b["speaker"] for b in batch],
    }
