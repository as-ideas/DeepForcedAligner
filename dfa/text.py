from typing import List


class Tokenizer:

    def __init__(self, symbols: List[str], pad_token='_') -> None:
        self.symbols = symbols
        self.idx_to_token = {i: s for i, s in enumerate(symbols, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.vocab_size = len(self.symbols) + 1

    def __call__(self, sentence):
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        return sequence

    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token])
