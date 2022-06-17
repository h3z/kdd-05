from typing import List

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from config.config import global_config
from utils import DATA_SPLIT_SIZE, DAY


def split_one(df: pd.DataFrame) -> List[pd.DataFrame]:
    # val_size = DATA_SPLIT_SIZE["val_size"] * 2
    # test_size = DATA_SPLIT_SIZE["test_size"]

    val_size = 25 * DAY
    test_size = 50 * DAY

    tscv = TimeSeriesSplit(n_splits=3, test_size=test_size)

    train_df, val_df, test_df = [], [], []
    for train, test in tscv.split(df):
        train_df.append(df.iloc[train[:-val_size]])
        val_df.append(df.iloc[train[-val_size:]])
        test_df.append(df.iloc[test[: -10 * DAY]])

    return train_df, val_df, test_df


# 223 train, 16 val, 15 test （实际只有 10 test）
def split_one2(df: pd.DataFrame) -> List[pd.DataFrame]:

    train_size = DATA_SPLIT_SIZE["train_size"]
    val_size = DATA_SPLIT_SIZE["val_size"]
    test_size = DATA_SPLIT_SIZE["test_size"]

    borders1 = [
        0,
        0 + train_size - global_config.input_timesteps,
        0 + train_size + val_size - global_config.input_timesteps,
    ]
    borders2 = [
        0 + train_size,
        0 + train_size + val_size,
        0 + train_size + val_size + test_size,
    ]

    res = []
    for i in range(3):
        res.append([df[borders1[i] : borders2[i]]])
    return res


def split(df: pd.DataFrame) -> List[pd.DataFrame]:
    if global_config.data_version == "all_turbines":
        res, trains, vals, tests = [], [], [], []
        for i in df.id.unique():
            train, val, test = split_one(df.query("id == @i"))
            res.append((train, val, test))
        return pd.concat(trains), pd.concat(vals), pd.concat(tests)
    else:
        return split_one(df)
