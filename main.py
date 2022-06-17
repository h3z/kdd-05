import datetime
import os
import pickle
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


def train_val(model_app, train_ds, val_ds):
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
            c.on_epoch_end(epoch, train_loss, val_loss, model_app) for c in callbacks
        ]
        if False in res:
            break

    [c.on_train_finish(model_app) for c in callbacks]


def model_path(location):
    turbine_id = global_config.turbine
    try:
        name = str(
            next(
                Path(global_config.checkpoints_dir).glob(
                    # "second_best__turbine_all_cuda_1__epoch_*.pt.fix"
                    # -------- exp/attn-seq2seq/1/ second_best_turbine_37__10.pt --------
                    # f"second_best_turbine_{turbine_id}__*.pt"
                    # -------- exp/attn-seq2seq/2/ second_best_turbine_all_cuda_0__4.pt.fix --------
                    # f"second_best_turbine_all_cuda_0__4.pt.fix"
                    # -------- exp/attn-seq2seq/3/ second_best__turbine_9___epoch_3.pt --------
                    f"second_best__turbine_{turbine_id}___epoch_*.pt"
                    # -------- exp/transformer/2/ second_best__turbine_all_cuda_0__epoch_7.pt.fix --------
                    # f"second_best__turbine_all_cuda_0__epoch_*.pt.fix"
                    # f"second_best__turbine_all_cuda_1__epoch_*.pt.fix"
                    # -------- exp/transformer/3/ second_best__turbine_col_0___epoch_9.pt --------
                    # f"second_best__turbine_col_{location.query('TurbID == @turbine_id').col.values[0]}___epoch_*.pt"
                    # -------- exp/transformer/loss-1/ second_best__turbine_col_0_cuda_0__epoch_10.pt.fix --------
                    # f"second_best__turbine_col_{dr.location.query('TurbID == @turbine_id').col.values[0]}_cuda_0__epoch_*.pt.fix"
                )
            )
        )
    except:
        print("skip")
        return -1, -1, -1, -1

    print(f"load model: {name}")
    return name


def compute_score(test_df, origin_test_df, location, test_preds):
    # test_gts = processor.postprocess(test_gts)[..., -1:]
    test_gts = np.concatenate(
        [
            batch[1].cpu().detach().numpy()
            for batch in data_loader.DataLoader(origin_test_df, location).get()
        ]
    )

    test_df = test_df.rename(columns=utils.to_origin_names)
    pickle.dump(
        (test_preds, test_gts),
        open(
            f"{global_config.checkpoints_dir}/test_preds_{global_config.turbine}.pkl",
            "wb",
        ),
    )

    # utils.wandb_plot(train_pred_records, val_pred_records, test_preds, test_gts)

    rmse, mae, score = evaluate([test_preds], [test_gts], [test_df])
    return rmse, mae, score


def prepare_data():
    dr = data_reader.DataReader(global_config.turbine)
    df = dr.train
    location = dr.location

    # split
    train_dfs, val_dfs, test_dfs = data_split.split(df)

    return train_dfs, val_dfs, test_dfs, location


def cv_i(ck_dir, i, train_dfs, val_dfs, test_dfs, location, args):
    global_config.checkpoints_dir = f"{ck_dir}/cv_{i}"
    os.makedirs(global_config.checkpoints_dir, exist_ok=True)

    if args.train:

        train_df = train_dfs[i]
        val_df = val_dfs[i]
        test_df = test_dfs[i]
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
        train_ds = data_loader.DataLoader(train_df, location, is_train=True).get()
        val_ds = data_loader.DataLoader(val_df, location).get()

        model_app = models.get(len(train_ds))

        train_val(model_app, train_ds, val_ds)
        # 下边这部分基于自己的 test_df 打分，其实没必要，所以只放在训练时候顺便看一眼，先留着看看吧
        test_ds = data_loader.DataLoader(test_df, location).get()
        test_preds, _ = train.predict(model_app, test_ds)
        test_preds = processor.postprocess(test_preds).squeeze()
        rmse, mae, score = compute_score(test_df, origin_test_df, location, test_preds)
        print("score", score)
    else:
        name = model_path(location)
        print(name)
        model_app = models.get()
        model_app.load_checkpoint(torch.load(name, map_location="cuda"))

    return model_app


def turbine_i(args, turbine_id) -> BaseModelApp:
    global_config.init_wandb()

    global_config.turbine = (
        turbine_id if global_config.data_version != "all_turbines" else None
    )
    print(global_config)

    # read & split csv
    train_dfs, val_dfs, test_dfs, location = prepare_data()

    cv_models = []
    test_preds_cvs = []
    ck_dir = global_config.checkpoints_dir
    for i in range(len(train_dfs)):
        m = cv_i(ck_dir, i, train_dfs, val_dfs, test_dfs, location, args)
        cv_models.append(m)

    test_df = test_dfs[-1]
    train_df = train_dfs[-1]
    origin_test_df = test_df.copy()
    # 现在开始不希望处理数据时依赖训练集了。所以以后 scaler 尽量不要用 all_col，直接用之前计算好的常数把。（有新特征的话再重新计算）
    processor = data_process.DataProcess(train_df)
    test_df = processor.preprocess(test_df)
    test_ds = data_loader.DataLoader(test_df, location).get()

    global_config.checkpoints_dir = ck_dir
    for m in cv_models:
        test_preds, _ = train.predict(m, test_ds)
        test_preds = processor.postprocess(test_preds).squeeze()
        test_preds_cvs.append(test_preds)

    rmse, mae, score = compute_score(
        test_df, origin_test_df, location, np.mean(test_preds_cvs, axis=0)
    )
    print("score", score)
    global_config.log({"rmse": rmse, "mae": mae, "score": score})

    global_config.wandb_finish()

    return rmse, mae, score


def main():
    print(datetime.datetime.now())
    args = utils.prep_env()

    scores = np.zeros((args.capacity_to - args.capacity_from, 3))
    for i in range(args.capacity_from + 1, args.capacity_to + 1):
        print(">>>>>>>>>>>>>> turbine", i, "<<<<<<<<<<<<<<<<<<")
        rmse, mae, score = turbine_i(args, i)
        scores[i - args.capacity_from - 1] = [rmse, mae, score]

    ########
    print(list(range(args.capacity_from + 1, args.capacity_to + 1)))
    print(f"rmse: \n{scores[:, 0]} \nmae: \n{scores[:, 1]} \nscore: \n{scores[:, 2]}")
    print(
        list(
            zip(list(range(args.capacity_from + 1, args.capacity_to + 1)), scores[:, 2])
        )
    )
    print(
        f"rmse: {scores[:, 0].mean()}, mae: {scores[:, 1].mean()}, score: {scores[:, 2].mean()}"
    )


if __name__ == "__main__":
    main()
