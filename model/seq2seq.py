import torch
from torch import nn
from model.base_model import BaseModelApp
import wandb
from train import optimizers, schedulers
import random


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.hidden_size = wandb.config.hidden_size
        self.num_layer = wandb.config.num_layer
        self.gru = nn.GRU(
            10,
            self.hidden_size,
            num_layers=self.num_layer,
            batch_first=True,
        )

    def forward(self, input):
        batch = input.shape[0]
        hidden = torch.zeros(self.num_layer, batch, self.hidden_size).to("cuda")

        output, hidden = self.gru(input, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        hidden_size = wandb.config.hidden_size
        self.num_layer = wandb.config.num_layer
        self.gru = nn.GRU(1, hidden_size, num_layers=self.num_layer, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        batch = input.shape[0]
        output, hidden = self.gru(input.view(batch, 1, -1), hidden)
        output = self.out(output).squeeze(-1)
        return output, hidden


class ModelApp(BaseModelApp):
    def __init__(self, *models) -> None:
        self.encoder = models[0]
        self.decoder = models[1]
        self.enc_optimizer = optimizers.get(self.encoder)
        self.dec_optimizer = optimizers.get(self.decoder)
        self.enc_scheduler = None
        self.dec_scheduler = None

        self.teacher_forcing_ratio = wandb.config.teacher_forcing_ratio

    def zero_grad(self):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

    def step(self):
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        if self.enc_scheduler:
            self.enc_scheduler.step()
        if self.dec_scheduler:
            self.dec_scheduler.step()

    def forward(self, batch_x, batch_y, is_training=False):
        # timesteps
        batch_y = batch_y.permute(1, 0, 2)
        target_length = len(batch_y)

        encoder_outputs, encoder_hidden = self.encoder(batch_x)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = (
            random.random() < self.teacher_forcing_ratio and is_training
        )

        preds = []
        decoder_input = torch.zeros_like(batch_y[0])

        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = (
                batch_y[di] if use_teacher_forcing else decoder_output.detach()
            )
            preds.append(decoder_output[:, -1:])

        return torch.stack(preds).permute(1, 0, 2)

    def checkpoint(self):
        return [self.encoder.state_dict(), self.decoder.state_dict()]

    def load_checkpoint(self, checkpoint):
        self.encoder.load_state_dict(checkpoint[0])
        self.decoder.load_state_dict(checkpoint[1])

    def init_scheduler(self, batch_count):
        self.enc_scheduler = schedulers.get(self.enc_optimizer, batch_count)
        self.dec_scheduler = schedulers.get(self.dec_optimizer, batch_count)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
