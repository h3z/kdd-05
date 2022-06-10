import torch

import utils
from callback.callback import Callback
from config.config import global_config
from model.base_model import BaseModelApp


class CacheCheckpoints(Callback):
    def on_epoch_end(self, epoch, loss, val_loss, model: BaseModelApp) -> bool:
        torch.save(
            model.checkpoint(), global_config.model_file_name(suffix=f"_{epoch}")
        )
        return True
