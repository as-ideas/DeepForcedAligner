import argparse
from pathlib import Path
import numpy as np
import torch
import tqdm
from torch import optim
from itertools import groupby

from dfa.dataset import new_dataloader
from dfa.model import Aligner
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import read_config, unpickle_binary, to_device
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])
    symbols = unpickle_binary(paths.data_dir / 'symbols.pkl')

    if args.checkpoint:
        print(f'Restoring model from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model = Aligner.from_checkpoint(checkpoint)
        assert checkpoint['symbols'] == symbols, 'Symbols from data do not match symbols from model!'
        print(f'Restored model with step {model.get_step()}')
    else:
        model_path = paths.checkpoint_dir / 'latest_model.pt'
        if model_path.exists():
            print(f'Restoring model from checkpoint: {model_path}')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model = Aligner.from_checkpoint(checkpoint)
            assert checkpoint['symbols'] == symbols, 'Symbols from data do not match symbols from model!'
            print(f'Restored model with step {model.get_step()}')
        else:
            print(f'Initializing new model from config {args.config}')
            model = Aligner(n_mels=config['audio']['n_mels'],
                            num_symbols=len(symbols)+1,
                            **config['model'])
            optim = optim.Adam(model.parameters(), lr=1e-4)
            checkpoint = {'model': model.state_dict(), 'optim': optim.state_dict(),
                          'config': config, 'symbols': symbols}

    trainer = Trainer(paths=paths)
    for split_num in range(5):
        target = 'output'
        trainer.train(checkpoint, train_params=config['training'], split_num=split_num)

        model_path = paths.checkpoint_dir / f'best_model_{split_num}.pt'
        dur_target_dir, pred_target_dir = Path(target) / 'durations', Path(target) / 'predictions'
        dur_target_dir.mkdir(parents=True, exist_ok=True)
        pred_target_dir.mkdir(parents=True, exist_ok=True)

        print(f'Loading model from {model_path}')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        symbols = unpickle_binary(paths.data_dir / 'symbols.pkl')
        tokenizer = Tokenizer(symbols)
        dataloader = new_dataloader(dataset_path=paths.data_dir / f'val_dataset_{split_num}.pkl', mel_dir=paths.mel_dir,
                                    token_dir=paths.token_dir, batch_size=1)

        model = Aligner.from_checkpoint(checkpoint).eval().to(device)
        print(f'Loaded model with step {model.get_step()} on device: {device}')
        assert symbols == checkpoint['symbols'], 'Symbols from dataset do not match symbols from model checkpoint!'

        print(f'Performing STT model inference...')
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            tokens, mel, tokens_len, mel_len = to_device(batch, device)
            pred_batch = model(mel)
            for b in range(tokens.size(0)):
                this_mel_len = mel_len[b]
                pred = pred_batch[b, :this_mel_len, :]
                pred = torch.softmax(pred, dim=-1)
                pred = pred.detach().cpu().numpy()
                item_id = batch['item_id'][b]
                np.save(pred_target_dir / f'{item_id}.npy', pred, allow_pickle=False)

        print(f'Transkribing...')
        dataset = unpickle_binary(paths.data_dir / f'val_dataset_{split_num}.pkl')
        result = []
        for item in dataset:
            file_name = item['item_id'] + '.npy'
            token_file, pred_file = paths.token_dir / file_name, pred_target_dir / file_name
            tokens = np.load(str(token_file), allow_pickle=False)
            pred = np.load(str(pred_file), allow_pickle=False)
            mel_len = item['mel_len']
            pred = pred[:mel_len]
            pred_max = np.argmax(pred, axis=1)
            text = tokenizer.decode(tokens.tolist())
            pred_text = tokenizer.decode(pred_max.tolist())
            pred_text_collapsed = ''.join([k for k, g in groupby(pred_text.replace('_', '')) if k!=0])
            result.append((item['item_id'], pred_text_collapsed))

        with open(f'output/transkribed_{split_num}.csv', 'w+', encoding='utf-8') as f:
            for a, b in result:
                f.write(f'{a}|{b}\n')