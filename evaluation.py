import os
import sys

import numpy as np
import torch
import wandb

import utils
from config import config
from data import data_loader, data_process, data_reader, data_split
from model import models
from model.base_model import BaseModelApp
from train import train
from utils import evaluate
import datetime

utils.fix_random()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TMP:
    def __init__(self) -> None:
        self.t = datetime.datetime.now()

    def print(self, t):
        # print(datetime.datetime.now() - self.t, t)
        self.t = datetime.datetime.now()


t = TMP()


def turbine_i(i) -> BaseModelApp:

    t.print("start")
    tconf = config.conf
    tconf["turbine"] = i
    config.__wandb__["mode"] = "offline"
    wandb.init(config=tconf, **config.__wandb__)
    t.print("wandb")

    # read csv
    df = data_reader.DataReader().train.query("id == @i")
    t.print("read")

    # split
    train_df, _, test_df = data_split.split(df)
    origin_test_df = test_df.copy()

    # preprocess
    processor = data_process.DataProcess(train_df)
    test_df = processor.preprocess(test_df)

    # torch DataLoader
    test_ds = data_loader.DataLoader(test_df).get()
    t.print("ds")

    # model = models.get()
    model_app = models.get()

    model_app.load_checkpoint(torch.load(f"checkpoints/{i}.pt"))
    t.print("model")

    # predict
    test_preds, _ = train.predict(model_app, test_ds)
    t.print("pred")

    # post process
    test_preds = processor.postprocess(test_preds)[..., -1:]
    test_gts = (
        iter(data_loader.DataLoader(origin_test_df).get())
        .next()[1]
        .cpu()
        .detach()
        .numpy()
    )
    test_df = test_df.rename(columns=config.to_origin_names)

    rmse, mae, score = evaluate([test_preds], [test_gts], [test_df])
    t.print("score")

    wandb.finish()
    t.print("finish")

    return model_app


def main():
    args = sys.argv[1:]
    capacity = int(args[0]) if len(args) != 0 else 1

    for i in range(capacity):
        i += 1
        print(">>>>>>>>>>>>>> turbine", i, "<<<<<<<<<<<<<<<<<<")
        model = turbine_i(i)


if __name__ == "__main__":
    main()
