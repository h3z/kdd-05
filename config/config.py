import json

PROJECT_NAME = "kdd-05"
ONLINE = False
RANDOM_STATE = 42
import wandb


class Config:
    def __init__(self) -> None:
        self.wandb = False
        self.conf = {
            "~lr": 1e-4,
            "~batch_size": 128,
            "~epochs": 20,
            "~early_stopping_patience": 5,
            "~optimizer": "adam",
            "~loss": "mse",
            "scheduler": "linear",  # linear, cosine
            "warmup": 0.1,
            "teacher_forcing_ratio": 0.5,
            # data
            "data_version": "full",  # small, full
            "scaler": "all_col",  # each_col, all_col
            "truncate": 0.98,
            "input_size": 10,
            "input_timesteps": 144,
            "output_timesteps": 144 * 2,
            # model
            "model": "gru",  # gru, seq2seq, attn_seq2seq
            "hidden_size": 48,
            "num_layer": 1,
        }
        self.wandb_conf = {
            "project": PROJECT_NAME,
            "entity": "hzzz",
            # mkdir /wandb/PROJECT_NAME
            "dir": f"/wandb/{PROJECT_NAME}",
            "mode": "online" if ONLINE else "offline",
        }

    def init(self, exp_file: None, wandb: True):
        if exp_file:
            self.conf = json.load(open(exp_file))
        self.wandb = wandb

    def init_wandb(self):
        wandb.init(config=self.conf, **self.wandb_conf)

    def __getattr__(self, name: str):
        if self.wandb:
            return global_config.get(name)
        else:
            return self.conf[name]

    def __getitem__(self, key):
        return self.__getattr__(key)


global_config = Config()
