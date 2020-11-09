import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        return x


class Aligner(torch.nn.Module):

    def __init__(self,
                 n_mels: int,
                 num_symbols: int,
                 lstm_dim: int,
                 conv_dim: int) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, conv_dim, 5),
            BatchNormConv(conv_dim, conv_dim, 5),
            BatchNormConv(conv_dim, conv_dim, 5),
        ])
        self.rnn = torch.nn.LSTM(conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols)

    def forward(self, x):
        if self.train:
            self.step += 1
        for conv in self.convs:
            x = conv(x)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x

    def get_step(self):
        return self.step.data.item()

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> 'Aligner':
        config = checkpoint['config']
        symbols = checkpoint['symbols']
        model = Aligner(n_mels=config['audio']['n_mels'],
                        num_symbols=len(symbols) + 1,
                        **config['model'])
        model.load_state_dict(checkpoint['model'])
        return model


class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=True)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=True)
        return x


class TTSModel(torch.nn.Module):

    def __init__(self,
                 n_mels: int,
                 num_symbols: int,
                 lstm_dim: int,
                 conv_dim: int) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.prenet = PreNet(n_mels)
        self.rnn = torch.nn.LSTM(num_symbols + 128, lstm_dim, batch_first=True, bidirectional=False)
        self.lin = torch.nn.Linear(lstm_dim, n_mels)
        self.n_mels = n_mels

    def forward(self, x, mel):
        if self.train:
            self.step += 1

        device = next(self.parameters()).device
        mel_in = torch.cat([torch.zeros(x.size(0), 1, self.n_mels, device=device), mel[:, :-1, :]], dim=1)
        mel_in = self.prenet(mel_in)
        x = torch.cat([x, mel_in], dim=-1)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x

    def get_step(self):
        return self.step.data.item()

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> 'TTSModel':
        config = checkpoint['config']
        symbols = checkpoint['symbols']
        model = TTSModel(n_mels=config['audio']['n_mels'],
                        num_symbols=len(symbols) + 1,
                        **config['model'])
        model.load_state_dict(checkpoint['model_tts'])
        return model