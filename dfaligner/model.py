import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.text import TextProcessor

from .config import DFAlignerConfig
from .duration_extraction import extract_durations_with_dijkstra


class BatchNormConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        return x


class Aligner(pl.LightningModule):
    def __init__(
        self,
        config: DFAlignerConfig,
    ) -> None:
        super().__init__()
        if not isinstance(config, AlignerConfig):
            config = AlignerConfig(**config)
        self.config = config
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        self.sep = "--"
        self.text_processor = TextProcessor(config)
        conv_dim = self.config.model.conv_dim
        lstm_dim = self.config.model.lstm_dim
        n_mels = self.config.preprocessing.audio.n_mels
        num_symbols = len(self.text_processor.symbols)
        self.log_dir = os.path.join(
            self.config.training.logger.save_dir,
            self.config.training.logger.name,
        )
        self.longest_mel = None
        self.longest_tokens = None
        self.batch_size = (
            self.config.training.batch_size
        )  # this is declared explicitly so that auto_scale_batch_size works: https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html
        self.register_buffer("step", torch.tensor(1, dtype=torch.int))
        self.ctc_loss = nn.CTCLoss()
        self.save_hyperparameters()  # TODO: ignore=['specific keys'] - I should ignore some unnecessary/problem values
        self.convs = nn.ModuleList(
            [
                BatchNormConv(n_mels, conv_dim, 5),
                BatchNormConv(conv_dim, conv_dim, 5),
                BatchNormConv(conv_dim, conv_dim, 5),
            ]
        )
        self.rnn = torch.nn.LSTM(
            conv_dim, lstm_dim, batch_first=True, bidirectional=True
        )
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols)

    def forward(self, x):
        if self.train:
            self.step += 1
        for conv in self.convs:
            x = conv(x)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x

    def on_load_checkpoint(self, checkpoint):
        """Deserialize the checkpoint hyperparameters.
        Note, this shouldn't fail on different versions of pydantic anymore,
        but it will fail on breaking changes to the config. We should catch those exceptions
        and handle them appropriately."""
        self.config = AlignerConfig(**checkpoint["hyper_parameters"]["config"])

    def on_save_checkpoint(self, checkpoint):
        """Serialize the checkpoint hyperparameters"""
        checkpoint["hyper_parameters"]["config"] = self.config.model_checkpoint_dump()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), self.config.training.optimizer.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
            "monitor": "validation/loss",
        }

    def predict_step(self, batch, batch_idx):
        save_dir = Path(self.config.preprocessing.save_dir)
        sep = "--"
        tokens = batch["tokens"]
        mel = batch["mel"]
        mel_len = batch["mel_len"]
        pred_batch = self(mel)
        for b in range(tokens.size(0)):
            this_mel_len = mel_len[b]
            pred = pred_batch[b, :this_mel_len, :]
            pred = torch.softmax(pred, dim=-1)
            pred = pred.detach().cpu().numpy()
            basename = batch["basename"][b]
            speaker = batch["speaker"][b]
            language = batch["language"][b]
            np.save(
                save_dir
                / "duration"
                / sep.join([basename, speaker, language, "duration.npy"]),
                pred,
                allow_pickle=False,
            )

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss_from_batch(batch)
        self.log("training/loss", loss.item(), prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss_from_batch(batch)
        self.log("validation/loss", loss.item(), prog_bar=False)
        if batch_idx == 0:
            self._generate_plots(batch["mel"], batch["tokens"])
        return loss

    def _calculate_loss_from_batch(self, batch):
        tokens = batch["tokens"]
        mel = batch["mel"]
        tokens_len = batch["tokens_len"]
        mel_len = batch["mel_len"]
        pred = self(mel)
        pred = pred.transpose(0, 1).log_softmax(2)
        return self.ctc_loss(pred, tokens, mel_len, tokens_len)

    def _generate_plots(self, mel, tokens):
        if self.longest_mel is None:
            self.longest_mel = mel
            self.longest_tokens = tokens.detach().cpu()
        pred = self(self.longest_mel)[0].detach().cpu().softmax(dim=-1)
        durations = extract_durations_with_dijkstra(
            self.longest_tokens.squeeze(0).numpy(), pred.numpy()
        )
        pred_max = pred.max(1)[1].numpy().tolist()
        pred_text = self.text_processor.token_sequence_to_text(pred_max)
        target_text = self.text_processor.token_sequence_to_text(
            self.longest_tokens.squeeze().tolist()
        )
        target_duration_rep = "".join(
            c * durations[i]
            for i, c in enumerate(self.text_processor._tokenizer.tokenize(target_text))
        )
        tensorboard = self.logger.experiment
        tensorboard.add_text(
            "validation/prediction", f"    {pred_text}", global_step=self.global_step
        )
        tensorboard.add_text(
            "validation/target", f"    {target_text}", global_step=self.global_step
        )
        tensorboard.add_text(
            "validation/target_duration_rep",
            f"    {target_duration_rep}",
            global_step=self.global_step,
        )
