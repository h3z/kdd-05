import torch

from config.config import global_config as config


def custom1(output, target):
    loss = (
        torch.mean((output - target) ** 2) + torch.mean(torch.abs(output - target))
    ) / 2
    return loss


def get():
    if config["~loss"] == "mse":
        return torch.nn.MSELoss()
    elif config["~loss"] == "bce":
        return torch.nn.BCELoss()
    elif config["~loss"] == "custom1":
        return custom1
