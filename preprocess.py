import argparse
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, Union

import numpy as np
import tqdm

from dfa.audio import Audio
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import get_files, read_config, pickle_binary, read_metafile


class Preprocessor:
    """Performs mel extraction and tokenization and stores the resulting torch tensors."""
    
    def __init__(self, audio: Audio, tokenizer: Tokenizer,
                 paths: Paths, text_dict: Dict[str, str],
                 mel_dim_last=True) -> None:
        self.audio = audio
        self.paths = paths
        self.tokenizer = tokenizer
        self.text_dict = text_dict
        self.mel_dim_last = mel_dim_last
    
    def __call__(self, wav_path: Path) -> Dict[str, Union[str, int]]:
        item_id = wav_path.stem
        if self.paths.precomputed_mels:
            mel = np.load(self.paths.precomputed_mels / f'{item_id}.npy')
            if not self.mel_dim_last:
                mel = mel.T
                print('b')
        else:
            wav = self.audio.load_wav(wav_path)
            mel = self.audio.wav_to_mel(wav)

        np.save(self.paths.mel_dir / f'{item_id}.npy', mel, allow_pickle=False)
        text = self.text_dict[item_id]
        tokens = np.array(self.tokenizer(text)).astype(np.int32)
        np.save(self.paths.token_dir / f'{item_id}.npy', tokens, allow_pickle=False)
        return {'item_id': item_id, 'tokens_len': tokens.shape[0], 'mel_len': mel.shape[0]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', help='Points to the config file.', default='config.yaml')
    parser.add_argument('--mel_dim_last', action='store_true',
                        help='Set if precomputed mels have mel channels as last dimension.')
    parser.add_argument('--num_workers', '-w', metavar='N', type=int, default=cpu_count() - 1,
                        help='The number of worker threads to use for preprocessing')

    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])
    audio = Audio.from_config(config['audio'])
    
    print(f'Config: {args.config}\n'
          f'Target data directory: {paths.data_dir}')
    
    text_dict = read_metafile(paths.metadata_path)
    symbols = set()
    for text in text_dict.values():
        symbols.update(set(text))
    symbols = sorted(list(symbols))
    wav_files = get_files(paths.dataset_dir, extension='.wav')
    wav_files = [x for x in wav_files if x.stem in text_dict] # for filtering in the metadata
    tokenizer = Tokenizer(symbols)
    preprocessor = Preprocessor(audio=audio, tokenizer=tokenizer, paths=paths,
                                text_dict=text_dict, mel_dim_last=args.mel_dim_last)
    pool = Pool(processes=args.num_workers)
    mapper = pool.imap_unordered(preprocessor, wav_files)
    dataset = []
    for i, item in tqdm.tqdm(enumerate(mapper), total=len(wav_files)):
        dataset.append(item)
    
    pickle_binary(dataset, paths.data_dir / 'dataset.pkl')
    pickle_binary(symbols, paths.data_dir / 'symbols.pkl')
    print('Preprocessing done.')
