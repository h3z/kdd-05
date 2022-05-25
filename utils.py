import random
import tempfile
import time

import numpy as np
import torch

from config.config import RANDOM_STATE


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
        acc = 1 - rmse(preds[idx, -day_len:, -1], gts[idx, -day_len:, -1]) / (
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

    overall_mae, overall_rmse = regressor_detailed_scores(
        predictions, grounds, raw_data_lst, settings
    )

    overall_mae *= 134
    overall_rmse *= 134
    total_score = (overall_mae + overall_rmse) / 2

    print(
        f"RMSE: {overall_rmse:.3f}, MAE: {overall_mae:.3f}, SCORE:  {total_score:.3f}"
    )

    return overall_rmse, overall_mae, total_score


def ignore_zeros(predictions, grounds):
    """
    Desc:
        Ignore the zero values for evaluation
    Args:
        predictions:
        grounds:
    Returns:
        Predictions and ground truths
    """
    preds = predictions[np.where(grounds != 0)]
    gts = grounds[np.where(grounds != 0)]
    return preds, gts


def rse(pred, ground_truth):
    """
    Desc:
        Root square error
    Args:
        pred:
        ground_truth: ground truth vector
    Returns:
        RSE value
    """
    _rse = 0.0
    if len(pred) > 0 and len(ground_truth) > 0:
        _rse = np.sqrt(np.sum((ground_truth - pred) ** 2)) / np.sqrt(
            np.sum((ground_truth - ground_truth.mean()) ** 2)
        )
    return _rse


def corr(pred, gt):
    """
    Desc:
        Correlation between the prediction and ground truth
    Args:
        pred:
        gt: ground truth vector
    Returns:
        Correlation
    """
    _corr = 0.0
    if len(pred) > 0 and len(gt) > 0:
        u = ((gt - gt.mean(0)) * (pred - pred.mean(0))).sum(0)
        d = np.sqrt(((gt - gt.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
        _corr = (u / d).mean(-1)
    return _corr


def mae(pred, gt):
    """
    Desc:
        Mean Absolute Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAE value
    """
    _mae = 0.0
    if len(pred) > 0 and len(gt) > 0:
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def mse(pred, gt):
    """
    Desc:
        Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSE value
    """
    _mse = 0.0
    if len(pred) > 0 and len(gt) > 0:
        _mse = np.mean((pred - gt) ** 2)
    return _mse


def rmse(pred, gt):
    """
    Desc:
        Root Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        RMSE value
    """
    return np.sqrt(mse(pred, gt))


def mape(pred, gt):
    """
    Desc:
        Mean Absolute Percentage Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAPE value
    """
    _mape = 0.0
    if len(pred) > 0 and len(gt) > 0:
        _mape = np.mean(np.abs((pred - gt) / gt))
    return _mape


def mspe(pred, gt):
    """
    Desc:
        Mean Square Percentage Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSPE value
    """
    return np.mean(np.square((pred - gt) / gt)) if len(pred) > 0 and len(gt) > 0 else 0


def regressor_scores(prediction, gt):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        prediction:
        gt: ground truth vector
    Returns:
        A tuple of metrics
    """
    _mae = mae(prediction, gt)
    _rmse = rmse(prediction, gt)
    return _mae, _rmse


def turbine_scores(pred, gt, raw_data, examine_len, stride=1):
    """
    Desc:
        Calculate the MAE and RMSE of one turbine
    Args:
        pred: prediction for one turbine
        gt: ground truth
        raw_data: the DataFrame of one wind turbine
        examine_len:
        stride:
    Returns:
        The averaged MAE and RMSE
    """
    cond = (
        (raw_data["Patv"] <= 0) & (raw_data["Wspd"] > 2.5)
        | (raw_data["Pab1"] > 89)
        | (raw_data["Pab2"] > 89)
        | (raw_data["Pab3"] > 89)
        | (raw_data["Wdir"] < -180)
        | (raw_data["Wdir"] > 180)
        | (raw_data["Ndir"] < -720)
        | (raw_data["Ndir"] > 720)
    )
    maes, rmses = [], []
    cnt_sample, out_seq_len, _ = pred.shape
    for i in range(0, cnt_sample, stride):
        indices = np.where(~cond[i : out_seq_len + i])
        prediction = pred[i]
        prediction = prediction[indices]
        targets = gt[i]
        targets = targets[indices]
        _mae, _rmse = regressor_scores(
            prediction[-examine_len:] / 1000, targets[-examine_len:] / 1000
        )
        if _mae != _mae or _rmse != _rmse:
            continue
        maes.append(_mae)
        rmses.append(_rmse)
    avg_mae = np.array(maes).mean()
    avg_rmse = np.array(rmses).mean()
    return avg_mae, avg_rmse


def regressor_detailed_scores(predictions, gts, raw_df_lst, settings):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        predictions:
        gts: ground truth vector
        raw_df_lst:
        settings:
    Returns:
        A tuple of metrics
    """
    start_time = time.time()
    all_mae, all_rmse = [], []
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        gt = gts[i]
        raw_df = raw_df_lst[i]
        _mae, _rmse = turbine_scores(
            prediction, gt, raw_df, settings["output_len"], settings["stride"]
        )
        if settings["is_debug"]:
            end_time = time.time()
            print(
                "\nSpent time for evaluating the {}-th turbine is {} secs\n".format(
                    i, end_time - start_time
                )
            )
            start_time = end_time
        all_mae.append(_mae)
        all_rmse.append(_rmse)
    total_mae = np.array(all_mae).sum()
    total_rmse = np.array(all_rmse).sum()
    return total_mae, total_rmse


def regressor_metrics(pred, gt):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        pred:
        gt: ground truth vector
    Returns:
        A tuple of metrics
    """
    _mae = mae(pred, gt)
    _mse = mse(pred, gt)
    _rmse = rmse(pred, gt)
    # pred, gt = ignore_zeros(pred, gt)
    _mape = mape(pred, gt)
    _mspe = mspe(pred, gt)
    return _mae, _mse, _rmse, _mape, _mspe
