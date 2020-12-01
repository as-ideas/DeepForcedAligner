import argparse
import itertools
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
import os
from dfa.audio import Audio
from dfa.duration_extraction import extract_durations_with_dijkstra, extract_durations_beam
from dfa.model import Aligner
from dfa.text import Tokenizer
from dfa.utils import read_metafile
from dfa.utils import read_config
from dfa.paths import Paths

class Cutter:

    def __init__(self, audio, tokenizer, model, metadata):
        self.audio = audio
        self.tokenizer = tokenizer
        self.model = model
        self.metadata = metadata

    def __call__(self, snippets_dir, wav_path, out_path):
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        snippets_dir = Path(snippets_dir)
        wav_ids = [w.stem for w in snippets_dir.glob('**/*.wav')]
        id_texts = [(id, t) for id, t in self.metadata if id in wav_ids]
        tokens_list = [self.tokenizer(t) for _, t in id_texts]
        tokens_flat = list(itertools.chain.from_iterable(tokens_list))
        tokens_flat = np.array(tokens_flat)
        wav = self.audio.load_wav(wav_path)
        mel = torch.tensor(self.audio.wav_to_mel(wav)).unsqueeze(0)
        pred = self.model(mel)
        pred = torch.softmax(pred, dim=-1)[0].detach().cpu().numpy()
        durations = extract_durations_with_dijkstra(tokens_flat, pred)
        start = 0
        wav_start = 0
        margin_left = 256 * 10
        margin_right = 256 * 10

        for index, tokens in enumerate(tokens_list, 0):
            durs = durations[start:start + len(tokens)]
            wav_part_len = sum(durs) * audio.hop_length
            wav_part_len_trimmed = sum(durs[:-1]) * audio.hop_length
            left = max(0, wav_start - margin_left)
            right = wav_start + wav_part_len_trimmed + margin_right
            wav_part = wav[left:right]
            start += len(tokens)
            wav_start += wav_part_len
            sf.write(out_path/f'{id_texts[index][0]}.wav', wav_part, samplerate=audio.sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    args = parser.parse_args()
    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])
    checkpoint = torch.load('/Users/cschaefe/workspace/DeepForcedAligner/dfa_checkpoints/model_step_100k.pt', map_location=torch.device('cpu'))
    config = checkpoint['config']
    symbols = checkpoint['symbols']
    audio = Audio.from_config(config['audio'])
    tokenizer = Tokenizer(symbols)
    model = Aligner.from_checkpoint(checkpoint).eval()
    print(f'model step {model.get_step()}')

    metafile = '/Users/cschaefe/datasets/ASVoice4/metadata_clean.csv'
    data_dir = '/Users/cschaefe/datasets/ASVoice4'
    wav_dir = '/Users/cschaefe/Axel Springer SE/TTS Audio Service (OG) - Post Production files (incl. breathing)'
    out_dir = '/Users/cschaefe/datasets/ASVoice4_breathing_cutted'

    metadata = []
    with open(metafile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            l_split = l.split('|')
            metadata.append((l_split[0], l_split[-1].strip() + ' '))

    wav_dir = Path(wav_dir)
    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    snippet_dirs = next(os.walk(data_dir))[1]
    cutter = Cutter(audio, tokenizer, model, metadata)
    snippet_dirs = sorted(list(snippet_dirs))
    for i, snippet_dir in enumerate(snippet_dirs, 1):
        snippet_dir = data_dir / snippet_dir
        wav_name = Path(snippet_dir).stem[:7] # this is really custom
        #if wav_name != 'r_00019':
        #    continue
        print(f'Generate data for dir {i} / {len(snippet_dirs)}: {snippet_dir}')
        wav_path = wav_dir / f'{wav_name}.wav'
        cutter(snippet_dir, wav_path, out_path=out_dir)