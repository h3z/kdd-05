import json
from pathlib import Path

PROJECT_NAME = "kdd-06"
ONLINE = False
RANDOM_STATE = 42
import torch

import wandb

__WANDB_ONLINE__ = "online"
__WANDB_OFFLINE__ = "offline"
__WANDB_CLOSE__ = "close"


# 因为忘了关这个，被自己坑了好几次。 好不容易训练完了，发现跑的不是指定的exp_file，而是这个
# IS_DEBUG = False
IS_DEBUG = False
# 可以指定搜参基于哪个来
DEBUG_CONFIG = {
    "exp": "gru/tmp",
    "wandb": "close",
    "train": 1,
}


class Config:
    def __init__(self) -> None:
        # self.conf = {
        #     "~lr": 1e-4,
        #     "~batch_size": 128,
        #     "~epochs": 1,
        #     "~early_stopping_patience": 5,
        #     "~optimizer": "adam",
        #     "~loss": "mse", # mse, custom1
        #     "scheduler": "linear",  # linear, cosine
        #     "warmup": 0.1,
        #     "teacher_forcing_ratio": 0.5,
        #     # data
        #     "data_version": "full",  # small, full, all_turbines, col_turbines
        #     "col_turbine": 0, # 0, 1, 2, 3, 4, 5
        #     "scaler": "all_col",  # each_col, all_col
        #     "truncate": 0.98,
        #     "input_size": 10,
        #     "input_timesteps": 144,
        #     "output_timesteps": 144 * 2,
        #     # model
        #     "model": "transformer",  # gru, seq2seq, attn_seq2seq, transformer
        #     "hidden_size": 48,
        #     "num_layer": 1,
        #     "n_heads": 8,
        #     "dim_val": 64,
        # }
        self.conf = {
            "~lr": 0.0005,
            "~batch_size": 128,
            "~epochs": 20,
            "~early_stopping_patience": 3,
            "~optimizer": "adam",
            "~loss": "mse",
            "scheduler": "linear",
            "warmup": 0.1,
            "data_version": "full",
            "scaler": "all_turbines_all_col",
            "truncate": 0.98,
            "input_size": 10,
            "input_timesteps": 288,
            "output_timesteps": 288,
            "model": "gru",
            "hidden_size": 48,
            "num_layer": 2,
            "turbine": 1,
            "dropout": 0.0,
            "split": "",  # wandb 搜参改不了。
        }

        self.wandb_conf = {
            "project": PROJECT_NAME,
            "entity": "hzzz",
            # mkdir /wandb/PROJECT_NAME
            "dir": f"/wandb/{PROJECT_NAME}",
            "mode": __WANDB_CLOSE__,
        }

    def init(self, args, extra):

        self.conf = json.load(open(f"{args.exp_dir}/{args.exp}/config.json"))
        for i in range(len(extra)):
            k, v = extra[i].split("=")
            k = k.replace("--", "")
            self.conf[k] = type(self.conf[k])(v)
        self.wandb_conf["mode"] = args.wandb
        self.checkpoints_dir = f"{args.exp_dir}/{args.exp}/checkpoints"

        Path(self.checkpoints_dir).mkdir(exist_ok=True, parents=True)
        print("Checkpoints:", self.checkpoints_dir)

        try:
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(torch.distributed.get_rank())
            self.distributed = True
        except:
            self.distributed = False

    @property
    def wandb_enable(self):
        return self.wandb_conf["mode"] != __WANDB_CLOSE__

    @property
    def cuda_rank(self):
        if self.distributed:
            return torch.distributed.get_rank()
        return 0

    @property
    def features(self):
        fea_str = self.__getattr__("features")
        return fea_str.split(" ")

    def model_file_name(self, prefix="", suffix=""):
        turbine = "turbine_"
        if self.turbine:
            turbine += str(self.turbine)
        elif self.col_turbine >= 0:
            turbine = turbine + "col_" + str(self.col_turbine)
        else:
            turbine += "all"

        cuda = f"cuda_{self.cuda_rank}" if self.distributed else ""
        return f"{self.checkpoints_dir}/{prefix}_{turbine}_{cuda}_{suffix}.pt"

    def init_wandb(self):
        if self.wandb_conf["mode"] != __WANDB_CLOSE__:
            wandb.init(config=self.conf, **self.wandb_conf, reinit=True)

    def log(self, json, step=None):
        if self.wandb_enable:
            wandb.log(json, step)

    def wandb_finish(self):
        if self.wandb_enable:
            wandb.finish()

    def update(self, __dict__):
        self.conf.update(__dict__)
        # if self.wandb_enable:
        #     wandb.config.update(__dict__, allow_val_change=True)

    def __getattr__(self, name: str):
        try:
            return wandb.config[name]
        except:
            # print("wrong", name)
            return self.conf[name] if name in self.conf else None

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __str__(self) -> str:
        return "wandb: " + str(wandb.config) + "\n\nconf: " + str(self.conf)


global_config = Config()
