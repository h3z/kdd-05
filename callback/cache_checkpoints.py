import torch

import utils
from callback.callback import Callback
from config.config import global_config
from model.base_model import BaseModelApp


class CacheCheckpoints(Callback):
    def __init__(self) -> None:
        self.n_iter = 0

    def on_epoch_end(self, loss, val_loss, model: BaseModelApp) -> bool:
        torch.save(
            model.checkpoint(),
            f"{global_config.checkpoints_dir}/{global_config.turbine}_{global_config.cuda_rank}_{self.n_iter}.pt",
        )
        self.n_iter += 1
        return True
