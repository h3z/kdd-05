import random
import tempfile

import numpy as np
import torch

from config.config import RANDOM_STATE

# from wpf_baseline_gru import metrics
from wpf_baseline_gru import metrics


def fix_random():
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def mktemp(f):
    return f"{tempfile.mkdtemp()}/{f}"


def evaluate(predictions, grounds, raw_data_lst):

    settings = {
        "input_len": 144,
        "output_len": 288,
        "start_col": 3,
        "in_var": 10,
        "out_var": 1,
        "day_len": 144,
        "train_size": 153,
        "val_size": 16,
        "test_size": 15,
        "total_size": 184,
        "lstm_layer": 2,
        "dropout": 0.05,
        "num_workers": 5,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 3,
        "lr": 0.0001,
        "lr_adjust": "type1",
        "capacity": 1,
        "turbine_id": 0,
        "stride": 1,
        "is_debug": False,
    }

    preds = np.array(predictions)
    gts = np.array(grounds)
    preds = np.sum(preds, axis=0)
    gts = np.sum(gts, axis=0)

    day_len = settings["day_len"]
    day_acc = []
    for idx in range(0, preds.shape[0]):
        acc = 1 - metrics.rmse(preds[idx, -day_len:, -1], gts[idx, -day_len:, -1]) / (
            settings["capacity"] * 1000
        )
        if acc != acc:
            continue
        day_acc.append(acc)
    day_acc = np.array(day_acc).mean()
    print()
    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    # out_len = settings["output_len"]
    # mae, rmse = metrics.regressor_scores(predictions[:, -out_len:, :] / 1000, grounds[:, -out_len:, :] / 1000)

    overall_mae, overall_rmse = metrics.regressor_detailed_scores(
        predictions, grounds, raw_data_lst, settings
    )

    overall_mae *= 134
    overall_rmse *= 134

    print(
        f"RMSE: {overall_rmse:.3f}, MAE: {overall_mae:.3f}, Accuracy:  {day_acc * 100:.4f}%"
    )
    total_score = (overall_mae + overall_rmse) / 2
    print(f"{total_score:.5f}")

    return overall_rmse, overall_mae, total_score
