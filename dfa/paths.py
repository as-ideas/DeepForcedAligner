from pathlib import Path


class Paths:

    def __init__(self, data_dir: str, checkpoint_dir: str):
        self.data_dir = Path(data_dir)
        self.mel_dir = self.data_dir / 'mels'
        self.token_dir = self.data_dir / 'tokens'
        self.checkpoint_dir = Path(checkpoint_dir)
        self.create_dirs()

    def create_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mel_dir.mkdir(parents=True, exist_ok=True)
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
