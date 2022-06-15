import math

import torch
import torch.nn as nn
from torch import Tensor, nn

from config.config import global_config as config


class PositionalEncoder(nn.Module):
    """
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/utils.py
    """

    def __init__(
        self,
        dropout: float = 0.1,
        d_model: int = 512,
    ):

        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()
        max_seq_len = config.input_timesteps

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values
        # dependent on position and i
        position = torch.arange(max_seq_len).unsqueeze(1)
        exp_input = torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)

        div_term = torch.exp(
            exp_input
        )  # Returns a new tensor with the exponential of the elements of exp_input

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # torch.Size([target_seq_len, dim_val])

        pe = pe.unsqueeze(0).transpose(
            0, 1
        )  # torch.Size([target_seq_len, input_size, dim_val])

        # register that pe is not a model parameter
        # self.register_buffer("pe", pe)
        # FIXME: workaround for DDP issue: https://github.com/pytorch/pytorch/issues/68407
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))
        # self.pe = torch.nn.Parameter(pe, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """

        add = self.pe[: x.size(1), :].squeeze(1)

        x = x + add

        return self.dropout(x)
