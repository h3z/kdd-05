import torch

from config.config import global_config


def get(model: torch.nn.Module):
    if global_config["~optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=global_config["~lr"])
