import argparse
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tqdm

from dfa.dataset import new_dataloader
from dfa.duration_extraction import extract_durations_with_dijkstra, extract_durations_beam
from dfa.model import Aligner
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import read_config, to_device, unpickle_binary
from itertools import groupby


if __name__ == '__main__':

    with open('output/transkribed.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for l in lines:
        split = l.split('|')
        id, text, pred = split[0], split[1], split[2]
        if id == '211636403_002':
            print(f'{id} {text}')
            print(f'{id} {pred}')