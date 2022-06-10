import torch

import utils
from callback.callback import Callback
from config.config import global_config
from model.base_model import BaseModelApp


class CacheCheckpoints(Callback):
    def __init__(self) -> None:
        self.n_iter = 0

    def on_epoch_end(self, loss, val_loss, model: BaseModelApp) -> bool:
        turbine = (
            f"turbine_{global_config.turbine if global_config.turbine else 'all' }_"
        )
        cuda = f"cuda_{global_config.cuda_rank}_" if global_config.distributed else ""
        torch.save(
            model.checkpoint(),
            f"{global_config.checkpoints_dir}/{turbine}{cuda}{self.n_iter}.pt",
        )
        self.n_iter += 1
        return True
