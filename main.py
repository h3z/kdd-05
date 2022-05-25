import os
import sys

import torch
import wandb

import utils
from callback import early_stopping, wandb_callback
from config import config
from data import data_loader, data_process, data_reader, data_split
from model import models
from model.base_model import BaseModelApp
from train import losses, optimizers, schedulers, train
from utils import evaluate

utils.fix_random()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parameters():
    return config.conf


def turbine_i(i) -> BaseModelApp:
    tconf = get_parameters()
    tconf["turbine"] = i
    wandb.init(config=tconf, **config.__wandb__)
    print(wandb.config)

    # read csv
    df = data_reader.DataReader().train.query("id == @i")

    # split
    train_df, val_df, test_df = data_split.split(df)
    print(
        f"train size: {len(train_df)/config.DAY}, val size: {len(val_df)/config.DAY}, test size: {len(test_df)/config.DAY}"
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

    # model = models.get()
    model_app = models.get(len(train_ds))

    # train
    criterion = losses.get()
    callbacks = [early_stopping.EarlyStopping(), wandb_callback.WandbCallback()]

    train_pred_records = []
    val_pred_records = []
    for epoch in range(wandb.config["~epochs"]):
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

        train_pred_records.append((train_gts, train_preds))
        val_pred_records.append((val_gts, val_preds))

        res = [c.on_epoch_end(train_loss, val_loss, model_app) for c in callbacks]
        if False in res:
            print("Early stopping")
            break

    [c.on_train_finish(model_app) for c in callbacks]

    # predict
    test_preds, test_gts = train.predict(model_app, test_ds)

    # post process
    test_preds = processor.postprocess(test_preds)[..., -1:]
    test_gts = processor.postprocess(test_gts)[..., -1:]
    test_df = test_df.rename(columns=config.to_origin_names)

    # wandb_plot(train_pred_records, val_pred_records, test_preds, test_gts)

    rmse, mae, score = evaluate([test_preds], [test_gts], [test_df])
    wandb.log({"rmse": rmse, "mae": mae, "score": score})

    wandb.finish()

    return model_app


def main():
    args = sys.argv[1:]
    capacity = int(args[0])
    for i in range(capacity):
        i += 1
        print(">>>>>>>>>>>>>> turbine", i, "<<<<<<<<<<<<<<<<<<")
        model = turbine_i(i)
        torch.save(model.checkpoint(), f"checkpoints/{i}.pt")


def wandb_plot(train_pred_records, val_pred_records, test_preds, test_gts):
    # plot gt & pred (last window)
    for i in range(wandb.config.output_timesteps):
        plot_dict = {}
        last_window_idx = (-1, i, 0)
        plot_dict["test_gt"] = test_gts[last_window_idx]
        plot_dict["test_pred"] = test_preds[last_window_idx]
        plot_dict["gt"] = val_pred_records[0][0][last_window_idx]
        for j in range(5):
            p = len(val_pred_records) // 5 * j
            plot_dict[f"pred_{j}"] = val_pred_records[p][1][last_window_idx]
            plot_dict[f"train_gt_{j}"] = train_pred_records[p][0][last_window_idx]
            plot_dict[f"train_pred_{j}"] = train_pred_records[p][1][last_window_idx]

        wandb.log(plot_dict)

    for i in range(288):
        from_batch = i // wandb.config.output_timesteps * wandb.config.output_timesteps
        batch_pred_i = i % wandb.config.output_timesteps
        batch_window_idx = (from_batch, batch_pred_i, 0)
        plot_dict = {}
        plot_dict["concat_test_gt"] = test_gts[batch_window_idx]
        plot_dict["concat_test_pred"] = test_preds[batch_window_idx]
        plot_dict["concat_gt"] = val_pred_records[0][0][batch_window_idx]
        for j in range(5):
            p = len(val_pred_records) // 5 * j
            record = train_pred_records[p]
            plot_dict[f"concat_pred_{j}"] = val_pred_records[p][1][batch_window_idx]
            plot_dict[f"concat_train_gt_{j}"] = record[0][batch_window_idx]
            plot_dict[f"concat_train_pred_{j}"] = record[1][batch_window_idx]

        wandb.log(plot_dict)


if __name__ == "__main__":
    main()
