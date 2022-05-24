import numpy as np
import torch


class Callback:
    def __init__(self) -> None:
        pass

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        pass

    def on_train_finish(self, model: torch.nn.Module):
        pass
