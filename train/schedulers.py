import torch
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from config.config import global_config


def get(optimizer: torch.optim.Optimizer, batch_num: int):
    if batch_num == None or batch_num <= 0:
        return None

    epochs = global_config["~epochs"]
    warmup = global_config.warmup
    num_training_steps = int(epochs * batch_num)

    if global_config.scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, int(warmup * num_training_steps), num_training_steps
        )

    if global_config.scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, int(warmup * num_training_steps), num_training_steps
        )
