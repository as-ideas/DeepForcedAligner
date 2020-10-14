import argparse
import torch
from torch import optim
from dfa.model import Aligner
from dfa.paths import Paths
from dfa.utils import read_config, unpickle_binary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--model', '-m', help='Points to the a model file to restore.')

    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths(**config['paths'])
    symbols = unpickle_binary(paths.data_dir / 'symbols.pkl')
    model = Aligner(n_mels=config['audio']['n_mels'],
                    num_symbols=len(symbols)+1,
                    **config['model'])

    optim = optim.Adam(model.parameters())

    # save and load model
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'config': config,
        'symbols': symbols,

    }, '/tmp/model.pt')

    checkpoint = torch.load('/tmp/model.pt')
    print(checkpoint['symbols'])