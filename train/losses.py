import torch

from config.config import global_config as config


def custom1(output, target):
    loss = (
        torch.mean((output - target) ** 2) + torch.mean(torch.abs(output - target))
    ) / 2
    return loss


def custom2(output, target, weights):
    return (
        torch.mean(
            ((output - target) ** 2 + torch.abs(output - target))
            * weights.reshape(output.shape)
        )
        / 2
    )


def get():
    if config["~loss"] == "mse":
        return lambda pred, gt, w: torch.nn.MSELoss()(pred, gt)
    elif config["~loss"] == "bce":
        return lambda pred, gt, w: torch.nn.BCELoss()(pred, gt)
    elif config["~loss"] == "custom1":
        return lambda pred, gt, w: custom1(pred, gt)
    elif config["~loss"] == "custom2":
        return custom2
