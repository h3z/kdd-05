import datetime

import numpy as np
import torch

import utils
from callback import early_stopping, wandb_callback
from config.config import global_config
from data import data_loader, data_process, data_reader, data_split
from model import models
from model.base_model import BaseModelApp
from train import train
from utils import __NO_CACHE__, __SAVE_CACHE__, __USE_CACHE__, evaluate

utils.fix_random()


def turbine_i(args) -> BaseModelApp:
    global_config.init_wandb()
    try:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(torch.distributed.get_rank())
        global_config.distributed = True
    except:
        global_config.distributed = False

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

    model_app = models.get(len(train_ds))

    model_app.load_checkpoint(torch.load(f"{args.checkpoints}/1-fix.pt"))

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
    # all turbines
    global_config.data_version = "full"
    global_config.scaler = "constant"
    global_config.__mean__ = 68.12220790619038
    global_config.__std__ = 186.9209197502287

    scores = np.zeros((args.capacity_to - args.capacity_from, 3))
    for i in range(args.capacity_from + 1, args.capacity_to + 1):
        print(">>>>>>>>>>>>>> turbine", i, "<<<<<<<<<<<<<<<<<<")
        global_config.turbine = i
        model, rmse, mae, score = turbine_i(args)
        scores[i - args.capacity_from - 1] = [rmse, mae, score]
        # torch.save(model.checkpoint(), f"{args.checkpoints}/{i}.pt")

    ########
    print(f"rmse: \n{scores[:, 0]} \nmae: \n{scores[:, 1]} \nscore: \n{scores[:, 2]}")
    print(
        f"rmse: {scores[:, 0].mean()}, mae: {scores[:, 1].mean()}, score: {scores[:, 2].mean()}"
    )


if __name__ == "__main__":
    main()
