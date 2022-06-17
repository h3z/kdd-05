import numpy as np

from callback.callback import Callback
from config.config import global_config
from model.base_model import BaseModelApp


class ScoreCallback(Callback):
    def __init__(self, f) -> None:
        self.f = f

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):

        return True

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_epoch_end(self, epoch, loss, val_loss, model_app: BaseModelApp) -> bool:
        self.f(model_app, epoch)
        return True

    def on_train_finish(self, model):
        pass
