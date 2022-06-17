import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from config.config import global_config as config
from utils import feature_cols


def get_src_trg(
    sequence: torch.Tensor,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    """
    Generate the src (encoder input), trg (decoder input) and trg_y (the target)
    sequences from a sequence.
    Args:
        sequence: tensor, a 1D tensor of length n where
                n = encoder input length + target sequence length
        enc_seq_len: int, the desired length of the input to the transformer encoder
        target_seq_len: int, the desired length of the target sequence (the
                        one against which the model output is compared)
    Return:
        src: tensor, 1D, used as input to the transformer model
        trg: tensor, 1D, used as input to the transformer model
        trg_y: tensor, 1D, the target sequence against which the model output
            is compared when computing loss.
    """
    enc_seq_len = config.input_timesteps
    dec_seq_len = config.output_timesteps
    target_seq_len = config.output_timesteps
    assert (
        len(sequence) == enc_seq_len + target_seq_len
    ), "Sequence length does not equal (input length + target length)"

    # encoder input
    src = sequence[:enc_seq_len]

    # decoder input. As per the paper, it must have the same dimension as the
    # target sequence, and it must contain the last value of src, and all
    # values of trg_y except the last (i.e. it must be shifted right by 1)
    trg = sequence[enc_seq_len - 1 : len(sequence) - 1]

    assert (
        len(trg) == target_seq_len
    ), "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[-target_seq_len:]

    assert (
        len(trg_y) == target_seq_len
    ), "Length of trg_y does not match target sequence length"

    return (
        src,
        trg,
        trg_y.squeeze(-1),
    )  # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]
