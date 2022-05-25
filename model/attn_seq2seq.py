import random

import torch
import torch.nn.functional as F
import wandb
from torch import nn

from model import seq2seq


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.hidden_size = wandb.config.hidden_size
        self.num_layer = wandb.config.num_layer
        self.gru = nn.GRU(
            10, self.hidden_size, num_layers=self.num_layer, batch_first=True
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
        num_layer = wandb.config.num_layer
        self.gru = nn.GRU(1, hidden_size, num_layers=num_layer, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        batch = input.shape[0]
        output, hidden = self.gru(input.view(batch, 1, -1), hidden)
        output = self.out(output).squeeze(-1)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = wandb.config.hidden_size
        num_layer = wandb.config.num_layer
        output_size = 1

        self.attn = nn.Linear(
            self.hidden_size + output_size, wandb.config.input_timesteps
        )
        self.attn_combine = nn.Linear(self.hidden_size + output_size, self.hidden_size)
        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, num_layers=num_layer, batch_first=True
        )
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        attn = self.attn(torch.cat((input, hidden[0]), -1))
        attn_weights = F.softmax(attn, dim=-1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        output = torch.cat((input, attn_applied.squeeze(1)), -1)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output.unsqueeze(1), hidden)

        output = self.out(output).squeeze(-1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class ModelApp(seq2seq.ModelApp):
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
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = (
                batch_y[di] if use_teacher_forcing else decoder_output.detach()
            )
            preds.append(decoder_output[:, -1:])

        return torch.stack(preds).permute(1, 0, 2)
