import datetime
import json
import sys

import numpy as np
import torch

import utils
import wandb
from callback import early_stopping, wandb_callback
from config import config
from data import data_loader, data_process, data_reader, data_split
from model import models
from model.base_model import BaseModelApp
from train import losses, train
from utils import __NO_CACHE__, __SAVE_CACHE__, __USE_CACHE__, evaluate

utils.fix_random()


class TMP:
    def __init__(self) -> None:
        self.t = datetime.datetime.now()

    def print(self, t):
        print(datetime.datetime.now() - self.t, t)
        self.t = datetime.datetime.now()


t = TMP()


def turbine_i(settings, args) -> BaseModelApp:
    wandb.init(config=settings, **config.__wandb__)
    print(wandb.config)

    if args.cache == __USE_CACHE__:
        train_df, val_df, test_df, processor = utils.load_cache("cache.pkl")
    elif args.cache == __NO_CACHE__ or args.cache == __SAVE_CACHE__:
        # read csv
        df = data_reader.DataReader(settings["turbine"]).train

        # split
        train_df, val_df, test_df = data_split.split(df)
        origin_test_df = test_df.copy()
        print(
            f"train size: {len(train_df)/utils.DAY}, val size: {len(val_df)/utils.DAY}, test size: {len(test_df)/utils.DAY}"
        )

        # preprocess
        processor = data_process.DataProcess(train_df)
        train_df = processor.preprocess(train_df)
        val_df = processor.preprocess(val_df)
        test_df = processor.preprocess(test_df)

        if args.cache == __SAVE_CACHE__:
            # 会覆盖
            utils.save_cache((train_df, val_df, test_df, processor), "cache.pkl")

    # torch DataLoader
    train_ds = data_loader.DataLoader(train_df, is_train=True).get()
    val_ds = data_loader.DataLoader(val_df).get()
    test_ds = data_loader.DataLoader(test_df).get()

    # model = models.get()
    model_app = models.get(len(train_ds))

    # train
    criterion = losses.get()
    callbacks = [early_stopping.EarlyStopping(), wandb_callback.WandbCallback()]

    # train_pred_records = []
    # val_pred_records = []
    for epoch in range(wandb.config["~epochs"]):
        t.print("next")
        train_loss, train_gts, train_preds = train.epoch_train(
            model_app,
            train_ds,
            criterion,
            callbacks,
        )
        val_loss, val_gts, val_preds = train.epoch_val(
            model_app,
            val_ds,
            criterion,
            callbacks,
        )
        print(epoch, ": train_loss", train_loss, "val_loss", val_loss)

        # train_pred_records.append((train_gts, train_preds))
        # val_pred_records.append((val_gts, val_preds))

        res = [c.on_epoch_end(train_loss, val_loss, model_app) for c in callbacks]
        if False in res:
            break

    [c.on_train_finish(model_app) for c in callbacks]

    # predict
    test_preds, _ = train.predict(model_app, test_ds)

    # post process
    test_preds = processor.postprocess(test_preds)[..., -1:]
    # test_gts = processor.postprocess(test_gts)[..., -1:]
    test_gts = (
        iter(data_loader.DataLoader(origin_test_df).get())
        .next()[1]
        .cpu()
        .detach()
        .numpy()
    )
    test_df = test_df.rename(columns=utils.to_origin_names)

    # utils.wandb_plot(train_pred_records, val_pred_records, test_preds, test_gts)

    rmse, mae, score = evaluate([test_preds], [test_gts], [test_df])
    wandb.log({"rmse": rmse, "mae": mae, "score": score})

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

        scores[i - 1] = [rmse, mae, score]
        torch.save(model.checkpoint(), f"{args.checkpoints}/{i}.pt")

    print(f"rmse: \n{scores[:, 0]} \nmae: \n{scores[:, 1]} \nscore: \n{scores[:, 2]}")
    print(
        f"rmse: {scores[:, 0].mean()}, mae: {scores[:, 1].mean()}, score: {scores[:, 2].mean()}"
    )


if __name__ == "__main__":
    print(datetime.datetime.now())
    main()
