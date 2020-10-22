import argparse

import numpy as np
import torch

from dfa.audio import Audio
from dfa.duration_extraction import extract_durations_with_dijkstra, extract_durations_beam
from dfa.model import Aligner
from dfa.text import Tokenizer
from dfa.utils import read_metafile
from dfa.utils import read_config
from dfa.paths import Paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    args = parser.parse_args()
    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])
    checkpoint = torch.load('/Volumes/data/logs/dfa/latest_model.pt', map_location=torch.device('cpu'))
    config = checkpoint['config']
    symbols = checkpoint['symbols']
    audio = Audio.from_config(config['audio'])
    tokenizer = Tokenizer(symbols)
    model = Aligner.from_checkpoint(checkpoint).eval()
    print(f'model step {model.get_step()}')
    
    main_dir = paths.dataset_dir
    text_dict = read_metafile(paths.metadata_path)
    file_id = list(text_dict.keys())[0]
    text = text_dict[file_id]
    
    target = np.array(tokenizer(text))
    
    mel = np.load((paths.mel_dir / file_id).with_suffix('.npy'))
    mel = torch.tensor(mel).float().unsqueeze(0)
    
    pred = model(mel)
    
    pred_max = pred[0].max(1)[1].detach().cpu().numpy().tolist()
    pred_text = tokenizer.decode(pred_max)
    pred = torch.softmax(pred, dim=-1)
    pred = pred.detach()[0].numpy()
    
    target_len = target.shape[0]
    pred_len = pred.shape[0]
    
    durations = extract_durations_with_dijkstra(target, pred)
    durations_beam, sequences = extract_durations_beam(target, pred, 5)
    expanded_string = ''.join([text[i] * dur for i, dur in enumerate(list(durations))])
    print(text)
    print(pred_text)
    print(expanded_string)
    print(tokenizer.decode(target[sequences[0][0]]))
    print(tokenizer.decode(target[sequences[-1][0]]))
    print(durations)
    print(durations_beam[0])
    print(durations_beam[-1])
    