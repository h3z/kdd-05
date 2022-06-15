import torch

from config.config import global_config as config


def get():
    if config["~loss"] == "mse":
        return torch.nn.MSELoss()
    elif config["~loss"] == "bce":
        return torch.nn.BCELoss()
