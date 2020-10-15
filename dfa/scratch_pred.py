from pathlib import Path

import torch

from dfa.audio import Audio
from dfa.model import Aligner
from dfa.text import Tokenizer

if __name__ == '__main__':

    checkpoint = torch.load('/Users/cschaefe/dfa_checkpoints/latest_model.pt', map_location=torch.device('cpu'))
    config = checkpoint['config']
    symbols = checkpoint['symbols']
    audio = Audio(**config['audio'])
    tokenizer = Tokenizer(symbols)
    model = Aligner.from_checkpoint(checkpoint).eval()
    print(f'model step {model.get_step()}')

    main_dir = Path('/Users/cschaefe/datasets/audio_data/Cutted_merged')
    file_id = '04902'

    wav = audio.load_wav(main_dir / f'{file_id}.wav')
    mel = audio.wav_to_mel(wav)
    mel = torch.tensor(mel).float().unsqueeze(0)

    pred = model(mel)

    pred[:, :, 0] = -9999 # remove pad pred
    pred = pred[0].max(1)[1].detach().cpu().numpy().tolist()
    pred_text = tokenizer.decode(pred)

    print(f'pred: {pred_text}')

