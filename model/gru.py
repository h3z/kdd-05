import torch
import wandb
from torch import nn


class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = wandb.config.hidden_size
        self.num_layer = wandb.config.num_layer
        self.output_timesteps = wandb.config.output_timesteps
        self.input_size = wandb.config.input_size

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layer,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.05)
        self.projection = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        batch = input.shape[0]
        x = torch.zeros([batch, self.output_timesteps, input.shape[2]]).to("cuda:0")
        x = torch.concat((input, x), 1)

        hidden = torch.zeros(self.num_layer, batch, self.hidden_size).to("cuda")
        output, hidden = self.gru(x, hidden)
        # output = self.dropout(output)
        output = self.projection(output)

        return output[:, -self.output_timesteps :, -1:]
