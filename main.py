import datetime
from pathlib import Path

import numpy as np
import torch

import utils
from callback import cache_checkpoints, early_stopping, wandb_callback
from config.config import global_config
from data import data_loader, data_process, data_reader, data_split
from model import models
from model.base_model import BaseModelApp
from train import train
from utils import __NO_CACHE__, __SAVE_CACHE__, __USE_CACHE__, evaluate

utils.fix_random()


def turbine_i(args) -> BaseModelApp:

    global_config.init_wandb()
    print(global_config)

    if args.cache == __NO_CACHE__ or args.cache == __SAVE_CACHE__:
        # read csv
        df = data_reader.DataReader(global_config.turbine).train

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

        # torch DataLoader
        train_ds = data_loader.DataLoader(train_df, is_train=True).get()
        val_ds = data_loader.DataLoader(val_df).get()
        test_ds = data_loader.DataLoader(test_df).get()

        if args.cache == __SAVE_CACHE__:
            # 会覆盖
            utils.save_cache(
                (test_df, processor, train_ds, val_ds, test_ds), "cache.pkl"
            )

    elif args.cache == __USE_CACHE__:
        test_df, processor, train_ds, val_ds, test_ds = utils.load_cache("cache.pkl")

    model_app = models.get(len(train_ds))
    if args.train:
        model_app.load_pretrained_params()
        # train
        callbacks = [
            cache_checkpoints.CacheCheckpoints(),
            early_stopping.EarlyStopping(),
            wandb_callback.WandbCallback(),
        ]

        # train_pred_records = []
        # val_pred_records = []
        for epoch in range(global_config["~epochs"]):
            train_losses, train_gts, train_preds = train.epoch_train(
                model_app,
                train_ds,
                callbacks,
            )
            val_losses, val_gts, val_preds = train.epoch_val(
                model_app,
                val_ds,
                callbacks,
            )

            train_loss = torch.stack(train_losses).mean().item()
            val_loss = torch.stack(val_losses).mean().item()
            print(epoch, ": train_loss", train_loss, "val_loss", val_loss)

            # train_pred_records.append((train_gts, train_preds))
            # val_pred_records.append((val_gts, val_preds))

            res = [
                c.on_epoch_end(epoch, train_loss, val_loss, model_app)
                for c in callbacks
            ]
            if False in res:
                break

        [c.on_train_finish(model_app) for c in callbacks]
    else:
        name = str(
            next(
                Path(global_config.checkpoints_dir).glob(
                    #f"best__turbine_{global_config.turbine}___epoch_*.pt"
                     f"second_best__turbine_{global_config.turbine}___epoch_*.pt"
                )
            )
        )

        print(f"load model: {name}")
        model_app.load_checkpoint(torch.load(name, map_location="cuda"))

    # predict
    test_preds, _ = train.predict(model_app, test_ds)

    # post process
    test_preds = processor.postprocess(test_preds)[..., -1:]
    # test_gts = processor.postprocess(test_gts)[..., -1:]
    test_gts = np.concatenate(
        [
            batch[1].cpu().detach().numpy()
            for batch in data_loader.DataLoader(origin_test_df).get()
        ]
    )
    test_df = test_df.rename(columns=utils.to_origin_names)

    # utils.wandb_plot(train_pred_records, val_pred_records, test_preds, test_gts)

    rmse, mae, score = evaluate([test_preds], [test_gts], [test_df])
    global_config.log({"rmse": rmse, "mae": mae, "score": score})

    global_config.wandb_finish()

    return model_app, rmse, mae, score


def main():
    print(datetime.datetime.now())
    args = utils.prep_env()

    scores = np.zeros((args.capacity_to - args.capacity_from, 3))
    for i in range(args.capacity_from + 1, args.capacity_to + 1):
        print(">>>>>>>>>>>>>> turbine", i, "<<<<<<<<<<<<<<<<<<")
        global_config.turbine = (
            i if global_config.data_version != "all_turbines" else None
        )

        model, rmse, mae, score = turbine_i(args)
        scores[i - args.capacity_from - 1] = [rmse, mae, score]

    ########
    print(f"rmse: \n{scores[:, 0]} \nmae: \n{scores[:, 1]} \nscore: \n{scores[:, 2]}")
    print(
        f"rmse: {scores[:, 0].mean()}, mae: {scores[:, 1].mean()}, score: {scores[:, 2].mean()}"
    )


if __name__ == "__main__":
    main()
