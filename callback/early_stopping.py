import numpy as np
import torch

import utils
from callback.callback import Callback
from config.config import global_config
from model.base_model import BaseModelApp


class EarlyStopping(Callback):
    def __init__(self) -> None:
        self.patience = global_config["~early_stopping_patience"]
        self.min_loss = np.inf
        self.counter = 0
        self.best_state_dict = None

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_epoch_end(self, loss, val_loss, model: BaseModelApp) -> bool:
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
            self.best_state_dict = model.checkpoint()
        else:
            self.counter += 1

        stop = self.counter >= self.patience
        if stop:
            print("Early stopping")
        return not stop

    def on_train_finish(self, model: BaseModelApp):
        f_name = f"{global_config.checkpoints_dir}/{global_config.turbine}"
        torch.save(
            model.checkpoint(),
            f_name
            + f"_{torch.distributed.get_rank() if global_config.distributed else 0}_last.pt",
        )

        model.load_checkpoint(self.best_state_dict)
        f = utils.mktemp("best_model.pth")
        torch.save(self.best_state_dict, f)
