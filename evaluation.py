import json

import numpy as np
import torch

import utils
import wandb
from config import config
from data import data_loader, data_process, data_reader, data_split
from model import models
from model.base_model import BaseModelApp
from train import train
from utils import evaluate

utils.fix_random()


def turbine_i(settings, args) -> BaseModelApp:
    config.__wandb__["mode"] = "offline"
    wandb.init(config=settings, **config.__wandb__)
    print(wandb.config)

    # read csv
    df = data_reader.DataReader().train.query(f"id == {settings['turbine']}")

    # split
    train_df, _, test_df = data_split.split(df)
    origin_test_df = test_df.copy()

    # preprocess
    processor = data_process.DataProcess(train_df)
    test_df = processor.preprocess(test_df)

    # torch DataLoader
    test_ds = data_loader.DataLoader(test_df).get()

    # model = models.get()
    model_app = models.get()

    model_app.load_checkpoint(
        torch.load(f"{args.checkpoints}/{settings['turbine']}.pt")
    )

    # predict
    test_preds, _ = train.predict(model_app, test_ds)

    # post process
    test_preds = processor.postprocess(test_preds)[..., -1:]
    test_gts = (
        iter(data_loader.DataLoader(origin_test_df).get())
        .next()[1]
        .cpu()
        .detach()
        .numpy()
    )
    test_df = test_df.rename(columns=utils.to_origin_names)

    rmse, mae, score = evaluate([test_preds], [test_gts], [test_df])

    wandb.finish()

    return model_app, rmse, mae, score


def main():
    args = utils.prep_env()
    settings = (
        json.load(open(args.exp_file)) if args.exp_file is not None else config.conf
    )

    scores = np.zeros((args.capacity, 3))
    for i in range(args.capacity):
        i += 1
        print(">>>>>>>>>>>>>> turbine", i, "<<<<<<<<<<<<<<<<<<")

        settings["turbine"] = i
        model, rmse, mae, score = turbine_i(settings, args)

        scores[i] = [rmse, mae, score]

    print(f"rmse: \n{scores[:, 0]} \nmae: \n{scores[:, 1]} \nscore: \n{scores[:, 2]}")
    print(
        f"rmse: {scores[:, 0].mean()}, mae: {scores[:, 1].mean()}, score: {scores[:, 2].mean()}"
    )


if __name__ == "__main__":
    main()
