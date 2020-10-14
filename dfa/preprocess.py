import argparse
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, Union

import torch
import tqdm

from dfa.audio import Audio
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import get_files, read_config, pickle_binary, read_metafile


class Preprocessor:
    """Performs mel extraction and tokenization and stores the resulting torch tensors."""

    def __init__(self, audio: Audio, tokenizer: Tokenizer,
                 paths: Paths, text_dict: Dict[str, str]) -> None:
        self.audio = audio
        self.paths = paths
        self.tokenizer = tokenizer
        self.text_dict = text_dict

    def __call__(self, wav_path: Path) -> Dict[str, Union[str, int]]:
        item_id = wav_path.stem
        wav = self.audio.load_wav(wav_path)
        mel = self.audio.wav_to_mel(wav)
        mel = torch.tensor(mel).float()
        text = self.text_dict[item_id]
        tokens = self.tokenizer(text)
        tokens = torch.tensor(tokens).long()
        torch.save(mel, self.paths.mel_dir / f'{item_id}.pt')
        torch.save(tokens, self.paths.token_dir / f'{item_id}.pt')
        return {'item_id': item_id, 'tokens_len': tokens.size(0), 'mel_len': mel.size(1)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--path', help='Points to the dataset path containing wavs and metafile.')
    parser.add_argument('--config', '-c', default='config.yaml', required=True, help='Points to the config file.')
    parser.add_argument('--num_workers', '-w', metavar='N', type=int, default=cpu_count() - 1,
                        help='The number of worker threads to use for preprocessing')
    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths(**config['paths'])
    audio = Audio(**config['audio'])

    print(f'Config: {args.config}\n'
          f'Target data directory: {paths.data_dir}')

    text_dict = read_metafile(args.path)
    symbols = set()
    for text in text_dict.values():
        symbols.update(set(text))
    symbols = sorted(list(symbols))

    wav_files = get_files(args.path, extension='.wav')
    tokenizer = Tokenizer(symbols)
    preprocessor = Preprocessor(audio=audio, tokenizer=tokenizer,
                                paths=paths, text_dict=text_dict)
    pool = Pool(processes=args.num_workers)
    mapper = pool.imap_unordered(preprocessor, wav_files)
    dataset = []
    for i, item in tqdm.tqdm(enumerate(mapper), total=len(wav_files)):
        dataset.append(item)

    pickle_binary(dataset, paths.data_dir / 'dataset.pkl')
    pickle_binary(symbols, paths.data_dir / 'symbols.pkl')
    print('Preprocessing done.')
