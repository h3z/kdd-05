import torch
import wandb


def get(model: torch.nn.Module):
    if wandb.config["~optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=wandb.config["~lr"])
