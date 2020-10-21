import argparse
import torch
from torch import optim
from torch.nn import CTCLoss

from dfa.dataset import new_dataloader
from dfa.model import Aligner
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import read_config, unpickle_binary, to_device
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--model', '-m', help='Points to the a model file to restore.')
    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])
    symbols = unpickle_binary(paths.data_dir / 'symbols.pkl')

    model = Aligner(n_mels=config['audio']['n_mels'],
                    num_symbols=len(symbols)+1,
                    **config['model'])
    optim = optim.Adam(model.parameters(), lr=1e-4)

    checkpoint = {'model': model.state_dict(), 'optim': optim.state_dict(),
                  'config': config, 'symbols': symbols}

    trainer = Trainer(paths=paths)
    trainer.train(checkpoint, train_params=config['training'])