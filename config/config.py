import json

PROJECT_NAME = "kdd-05"
ONLINE = False
RANDOM_STATE = 42
import wandb

__WANDB_ONLINE__ = "online"
__WANDB_OFFLINE__ = "offline"
__WANDB_CLOSE__ = "close"


class Config:
    def __init__(self) -> None:
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

    def init(self, args):
        if args.exp_file:
            self.conf = json.load(open(args.exp_file))
        self.wandb_conf["mode"] = args.wandb

    @property
    def wandb_enable(self):
        return self.wandb_conf["mode"] != __WANDB_CLOSE__

    def init_wandb(self):
        if self.wandb_enable:
            wandb.init(config=self.conf, **self.wandb_conf)

    def log(self, json):
        if self.wandb_enable:
            wandb.log(json)

    def wandb_finish(self):
        if self.wandb_enable:
            wandb.finish()

    def __getattr__(self, name: str):
        if self.wandb_enable:
            return wandb.config[name]
        else:
            return self.conf[name]

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __str__(self) -> str:
        return str(self.conf)


global_config = Config()
