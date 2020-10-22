from pathlib import Path

import numpy as np
import torch

from dfa.audio import Audio
from dfa.duration_extraction import extract_durations_with_dijkstra, extract_durations_beam
from dfa.model import Aligner
from dfa.text import Tokenizer
from dfa.utils import read_metafile

if __name__ == '__main__':

    checkpoint = torch.load('/Users/cschaefe/dfa_checkpoints/latest_model_old.pt', map_location=torch.device('cpu'))
    config = checkpoint['config']
    symbols = checkpoint['symbols']
    audio = Audio.from_config(config['audio'])
    tokenizer = Tokenizer(symbols)
    model = Aligner.from_checkpoint(checkpoint).eval()
    print(f'model step {model.get_step()}')

    main_dir = Path('/Users/cschaefe/datasets/audio_data/Cutted_merged')
    text_dict = read_metafile(main_dir)
    file_id = '04902'
    wav = audio.load_wav(main_dir / f'{file_id}.wav')
    text = text_dict[file_id]

    target = np.array(tokenizer(text))

    mel = audio.wav_to_mel(wav)
    mel = torch.tensor(mel).float().unsqueeze(0)

    pred = model(mel)

    pred_max = pred[0].max(1)[1].detach().cpu().numpy().tolist()
    pred_text = tokenizer.decode(pred_max)

    pred = torch.softmax(pred, dim=-1)
    pred = pred.detach()[0].numpy()

    target_len = target.shape[0]
    pred_len = pred.shape[0]

    durations = extract_durations_with_dijkstra(target, pred)
    sequences = extract_durations_beam(pred, target, 5)
    expanded_string = ''.join([text[i] * dur for i, dur in enumerate(list(durations))])
    print(text)
    print(pred_text)
    print(expanded_string)
    print(tokenizer.decode(target[sequences[0][0]]))
    print(tokenizer.decode(target[sequences[-1][0]]))
    print(durations)



