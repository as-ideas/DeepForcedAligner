import librosa
import numpy as np


class Audio:
    """Performs audio processing such as generating mel specs and normalization."""
    
    def __init__(self,
                 n_mels: int,
                 sample_rate: int,
                 hop_length: int,
                 win_length: int,
                 n_filters: int,
                 fmin: int,
                 fmax: int):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_filters = n_filters
        self.fmin = fmin
        self.fmax = fmax
    
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
    
    @staticmethod
    def normalize(mel):
        S = np.clip(mel, a_min=1.0e-5, a_max=None)
        return np.log(S)
    
    @staticmethod
    def denormalize(mel):
        return np.exp(mel)
    
    @classmethod
    def from_config(cls, config):
        return cls(
            sample_rate=config['sample_rate'],
            n_filters=config['n_filters'],
            n_mels=config['n_mels'],
            win_length=config['win_length'],
            hop_length=config['hop_length'],
            fmin=config['fmin'],
            fmax=config['fmax'])
