import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import torch

import utils
from callback import cache_checkpoints, early_stopping, score_callback, wandb_callback
from config.config import global_config
from data import data_loader, data_process, data_reader, data_split
from model import models
from model.base_model import BaseModelApp
from train import train
from utils import __NO_CACHE__, __SAVE_CACHE__, __USE_CACHE__, evaluate

utils.fix_random()


def _score_(test_ds, processor, origin_test_df, location):
    def f(model_app, step):
        test_preds, _ = train.predict(model_app, test_ds)
        test_preds = processor.postprocess(test_preds).squeeze()
        rmse, mae, score = compute_score(origin_test_df, location, test_preds)
        global_config.log({"score": score})
        print("score", score)

    return f


def train_val(model_app, train_ds, val_ds, _cb):
    model_app.load_pretrained_params()
    # train
    callbacks = [
        cache_checkpoints.CacheCheckpoints(),
        early_stopping.EarlyStopping(),
        wandb_callback.WandbCallback(),
        _cb,
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
    # "second_best__turbine_all_cuda_1__epoch_*.pt.fix"
    # -------- exp/attn-seq2seq/1/ second_best_turbine_37__10.pt --------
    # f"second_best_turbine_{turbine_id}__*.pt"
    # -------- exp/attn-seq2seq/2/ second_best_turbine_all_cuda_0__4.pt.fix --------
    # f"second_best_turbine_all_cuda_0__4.pt.fix"
    # -------- exp/attn-seq2seq/3/ second_best__turbine_9___epoch_3.pt --------
    name = f"second_best__turbine_{turbine_id}___epoch_*.pt"
    # -------- exp/transformer/2/ second_best__turbine_all_cuda_0__epoch_7.pt.fix --------
    # f"second_best__turbine_all_cuda_0__epoch_*.pt.fix"
    # f"second_best__turbine_all_cuda_1__epoch_*.pt.fix"
    # -------- exp/transformer/3/ second_best__turbine_col_0___epoch_9.pt --------
    # f"second_best__turbine_col_{location.query('TurbID == @turbine_id').col.values[0]}___epoch_*.pt"
    # -------- exp/transformer/loss-1/ second_best__turbine_col_0_cuda_0__epoch_10.pt.fix --------
    # f"second_best__turbine_col_{dr.location.query('TurbID == @turbine_id').col.values[0]}_cuda_0__epoch_*.pt.fix"
    print(global_config.checkpoints_dir, name)
    name = str(next(Path(global_config.checkpoints_dir).glob(name)))

    print(f"load model: {name}")
    return name


def compute_score(origin_test_df, location, test_preds):
    # test_gts = processor.postprocess(test_gts)[..., -1:]
    test_gts = np.concatenate(
        [
            batch[1].cpu().detach().numpy()
            for batch in data_loader.DataLoader(origin_test_df).get()
        ]
    )

    origin_test_df = origin_test_df.rename(columns=utils.to_origin_names)
    pickle.dump(
        (test_preds, test_gts),
        open(
            f"{global_config.checkpoints_dir}/test_preds_{global_config.turbine}.pkl",
            "wb",
        ),
    )

    # utils.wandb_plot(train_pred_records, val_pred_records, test_preds, test_gts)

    rmse, mae, score = evaluate([test_preds], [test_gts], [origin_test_df])
    return rmse, mae, score


def prepare_data():
    dr = data_reader.DataReader(global_config.turbine, global_config.col_turbine)

    # split
    train_dfs, val_dfs, test_dfs = data_split.split(dr.train)
    print("cv len", len(train_dfs))
    return train_dfs, val_dfs, test_dfs, dr.location


def cv_i(train_df, val_df, test_df, location, args):
    if args.train:
        global_config.init_wandb()
        print("train config:", global_config)
        origin_test_df = test_df.copy()

        # preprocess
        processor = data_process.DataProcess(train_df)
        train_df = processor.preprocess(train_df)
        val_df = processor.preprocess(val_df)
        test_df = processor.preprocess(test_df)

        # torch DataLoader
        train_ds = data_loader.DataLoader(train_df, is_train=True).get()
        val_ds = data_loader.DataLoader(val_df).get()
        test_ds = data_loader.DataLoader(test_df).get()

        model_app = models.get(len(train_ds))

        train_val(
            model_app,
            train_ds,
            val_ds,
            score_callback.ScoreCallback(
                _score_(test_ds, processor, origin_test_df, location)
            ),
        )
        global_config.wandb_finish()

    else:
        name = model_path(location)
        print(name)
        model_app = models.get()
        model_app.load_checkpoint(torch.load(name, map_location="cuda"))

    return model_app


def turbine_i(args, turbine_id) -> BaseModelApp:

    global_config.turbine = (
        turbine_id if global_config.data_version != "all_turbines" else None
    )
    print(global_config)

    # read & split csv
    train_dfs, val_dfs, test_dfs, location = prepare_data()

    cv_models = []
    test_preds_cvs = []
    ck_root_dir = global_config.checkpoints_dir
    for i in range(len(train_dfs)):
        global_config.checkpoints_dir = f"{ck_root_dir}/cv_{i}"
        global_config.update({"cv": i})
        os.makedirs(global_config.checkpoints_dir, exist_ok=True)

        m = cv_i(train_dfs[i], val_dfs[i], test_dfs[i], location, args)
        cv_models.append(m)

    test_df = test_dfs[-1]
    train_df = train_dfs[-1]
    origin_test_df = test_df.copy()
    # 现在开始不希望处理数据时依赖训练集了。所以以后 scaler 尽量不要用 all_col，直接用之前计算好的常数把。（有新特征的话再重新计算）
    processor = data_process.DataProcess(train_df)
    test_df = processor.preprocess(test_df)
    test_ds = data_loader.DataLoader(test_df).get()

    global_config.checkpoints_dir = ck_root_dir
    scores = []
    for m in cv_models:
        test_preds, _ = train.predict(m, test_ds)
        test_preds = processor.postprocess(test_preds).squeeze()
        test_preds_cvs.append(test_preds)
        scores.append(compute_score(origin_test_df, location, test_preds)[2])

    rmse, mae, score = compute_score(
        origin_test_df, location, np.mean(test_preds_cvs, axis=0)
    )
    scores.append(score)
    print("scores", scores)

    return rmse, mae, scores


def main():
    print(datetime.datetime.now())
    args = utils.prep_env()

    scores = []
    for i in range(args.capacity_from, args.capacity_to):
        turbine_id = i + 1
        print(">>>>>>>>>>>>>> turbine", turbine_id, "<<<<<<<<<<<<<<<<<<")
        rmse, mae, score = turbine_i(args, turbine_id)
        scores.append((turbine_id, *score))

    ########
    print(scores)
    print("score: ", np.array(scores)[:, -1].mean())


if __name__ == "__main__":
    main()
