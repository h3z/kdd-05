import torch

from config.config import global_config


def get():
    if global_config["~loss"] == "mse":
        return torch.nn.MSELoss()
    elif global_config["~loss"] == "bce":
        return torch.nn.BCELoss()
