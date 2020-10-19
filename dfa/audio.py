import sys

import librosa
import numpy as np


class Normalizer:
    def normalize(self, S):
        raise NotImplementedError
    
    def denormalize(self, S):
        raise NotImplementedError


class Audio:
    """Performs audio processing such as generating mel specs and normalization."""
    
    def __init__(self,
                 n_mels: int,
                 sample_rate: int,
                 hop_length: int,
                 win_length: int,
                 n_filters: int,
                 fmin: int,
                 fmax: int,
                 normalizer: Normalizer):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_filters = n_filters
        self.fmin = fmin
        self.fmax = fmax
        self.normalizer = normalizer
    
    def load_wav(self, path):
        wav, _ = librosa.load(path, sr=self.sample_rate)
        return wav
    
    def wav_to_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_filters,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax)
        mel = mel.T
        return self.normalize(mel)
    
    def normalize(self, mel):
        return self.normalizer.normalize(mel)
    
    def denormalize(self, mel):
        return self.normalizer.denormalize(mel)
    
    @classmethod
    def from_config(cls, config):
        normalizer = getattr(sys.modules[__name__], config['normalizer'])()
        return cls(
            sample_rate=config['sample_rate'],
            n_filters=config['n_filters'],
            n_mels=config['n_mels'],
            win_length=config['win_length'],
            hop_length=config['hop_length'],
            fmin=config['fmin'],
            fmax=config['fmax'],
            normalizer=normalizer)


class MelGAN(Normalizer):
    def __init__(self):
        super().__init__()
        self.clip_min = 1.0e-5
    
    def normalize(self, S):
        S = np.clip(S, a_min=self.clip_min, a_max=None)
        return np.log(S)
    
    def denormalize(self, S):
        return np.exp(S)


class WaveRNN(Normalizer):
    def __init__(self):
        super().__init__()
        self.min_level_db = - 100
        self.max_norm = 4
    
    def normalize(self, S):
        S = self.amp_to_db(S)
        S = np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)
        return (S * 2 * self.max_norm) - self.max_norm
    
    def denormalize(self, S):
        S = (S + self.max_norm) / (2 * self.max_norm)
        S = (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db
        return self.db_to_amp(S)
    
    def amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))
    
    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)
