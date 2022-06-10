import copy

import numpy as np
import torch

from callback.callback import Callback
from config.config import global_config
from model.base_model import BaseModelApp


class EarlyStopping(Callback):
    def __init__(self) -> None:
        self.patience = global_config["~early_stopping_patience"]
        self.min_loss = np.inf
        self.counter = 0
        self.best_state_dict = None
        self.best_epoch = 0

        self.second_min_loss = np.inf
        self.second_best_state_dict = None
        self.second_best_epoch = 0

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_epoch_end(self, epoch, loss, val_loss, model: BaseModelApp) -> bool:
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.checkpoint())
            self.best_epoch = epoch
        else:
            self.counter += 1

        if epoch > 1 and val_loss < self.second_min_loss:
            self.second_min_loss = val_loss
            self.second_best_state_dict = copy.deepcopy(model.checkpoint())
            self.second_best_epoch = epoch

        stop = self.counter >= self.patience
        if stop:
            print("Early stopping")
        return not stop

    def on_train_finish(self, model: BaseModelApp):
        model.load_checkpoint(self.best_state_dict)
        torch.save(
            self.best_state_dict,
            global_config.model_file_name(prefix="best_", suffix=f"_{self.best_epoch}"),
        )

        torch.save(
            self.second_best_state_dict,
            global_config.model_file_name(
                prefix="second_best_", suffix=f"_{self.second_best_epoch}"
            ),
        )
