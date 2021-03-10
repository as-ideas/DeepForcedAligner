import numpy as np
import torch
import tqdm
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from dfa.dataset import new_dataloader, get_longest_mel_id
from dfa.duration_extraction import extract_durations_with_dijkstra
from dfa.model import Aligner
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import to_device


class Trainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.ctc_loss = CTCLoss()

    def train(self, checkpoint: dict, train_params: dict, split_num) -> None:
        # Used for generating plots
        self.writer = SummaryWriter(log_dir=self.paths.checkpoint_dir / f'tensorboard_{split_num}')
        longest_id = get_longest_mel_id(dataset_path=self.paths.data_dir / f'val_dataset_{split_num}.pkl')
        self.longest_mel = np.load(str(self.paths.mel_dir / f'{longest_id}.npy'), allow_pickle=False)
        self.longest_tokens = np.load(str(self.paths.token_dir / f'{longest_id}.npy'), allow_pickle=False)
        lr = train_params['learning_rate']
        epochs = train_params['epochs']
        batch_size = train_params['batch_size']
        ckpt_steps = train_params['checkpoint_steps']
        plot_steps = train_params['plot_steps']

        config = checkpoint['config']
        symbols = checkpoint['symbols']
        tokenizer = Tokenizer(symbols)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = Aligner.from_checkpoint(checkpoint).to(device)
        optim = Adam(model.parameters())
        optim.load_state_dict(checkpoint['optim'])

        for g in optim.param_groups:
            g['lr'] = lr

        dataloader = new_dataloader(dataset_path=self.paths.data_dir / f'train_dataset_{split_num}.pkl', mel_dir=self.paths.mel_dir,
                                    token_dir=self.paths.token_dir, batch_size=batch_size)

        val_dataloader = new_dataloader(dataset_path=self.paths.data_dir / f'val_dataset_{split_num}.pkl', mel_dir=self.paths.mel_dir,
                                    token_dir=self.paths.token_dir, batch_size=8)

        loss_sum = 0.
        start_epoch = model.get_step() // len(dataloader)
        min_val_loss = 9999.

        for epoch in range(start_epoch + 1, epochs + 1):
            pbar = tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader))
            for i, batch in pbar:
                pbar.set_description(desc=f'Epoch: {epoch} | Step {model.get_step()} '
                                          f'| Loss: {loss_sum / i:#.4}', refresh=True)
                tokens, mel, tokens_len, mel_len = to_device(batch, device)

                pred = model(mel)
                pred = pred.transpose(0, 1).log_softmax(2)

                loss = self.ctc_loss(pred, tokens, mel_len, tokens_len)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                loss_sum += loss.item()

                self.writer.add_scalar('CTC_Loss/train', loss.item(), global_step=model.get_step())
                self.writer.add_scalar('Params/batch_size', batch_size, global_step=model.get_step())
                self.writer.add_scalar('Params/learning_rate', lr, global_step=model.get_step())

                if model.get_step() % ckpt_steps == 0:
                    torch.save({'model': model.state_dict(), 'optim': optim.state_dict(),
                                'config': config, 'symbols': symbols},
                               self.paths.checkpoint_dir / f'model_step_{model.get_step() // 1000}k.pt')

                if model.get_step() % plot_steps == 0:
                    self.generate_plots(model, tokenizer)

            val_loss = self.evaluate(model, val_dataloader)
            if val_loss < min_val_loss:
                print(f'Saving best model at step {model.get_step()} with val loss {val_loss}')
                min_val_loss = val_loss
                latest_checkpoint = self.paths.checkpoint_dir / f'best_model_{split_num}.pt'
                torch.save({'model': model.state_dict(), 'optim': optim.state_dict(),
                            'config': config, 'symbols': symbols},
                           latest_checkpoint)

            self.writer.add_scalar('CTC_Loss/val', val_loss, global_step=model.get_step())
            loss_sum = 0

    def evaluate(self, model: Aligner, dataloader):
        val_loss = 0.
        model.eval()
        device = next(model.parameters()).device
        for i, batch in enumerate(dataloader, 1):
            tokens, mel, tokens_len, mel_len = to_device(batch, device)
            with torch.no_grad():
                pred = model(mel)
            pred = pred.transpose(0, 1).log_softmax(2)
            loss = self.ctc_loss(pred, tokens, mel_len, tokens_len)
            val_loss += loss.item()
        model.train()
        return val_loss / len(dataloader)

    def generate_plots(self, model: Aligner, tokenizer: Tokenizer) -> None:
        model.eval()
        device = next(model.parameters()).device
        longest_mel = torch.tensor(self.longest_mel).unsqueeze(0).float().to(device)
        pred = model(longest_mel)[0].detach().cpu().softmax(dim=-1)
        durations = extract_durations_with_dijkstra(self.longest_tokens, pred.numpy())
        pred_max = pred.max(1)[1].numpy().tolist()
        pred_text = tokenizer.decode(pred_max)
        target_text = tokenizer.decode(self.longest_tokens)
        target_duration_rep = ''.join(c * durations[i] for i, c in enumerate(target_text))
        self.writer.add_text('Text/Prediction', '    ' + pred_text, global_step=model.get_step())
        self.writer.add_text('Text/Target_Duration_Repeated',
                             '    ' + target_duration_rep, global_step=model.get_step())
        self.writer.add_text('Text/Target', '    ' + target_text, global_step=model.get_step())
        model.train()
