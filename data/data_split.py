from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import global_config
from utils import DATA_SPLIT_SIZE


def split_one(df: pd.DataFrame) -> List[pd.DataFrame]:

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
        res.append(df[borders1[i] : borders2[i]])
    return res


def split(df: pd.DataFrame) -> List[pd.DataFrame]:
    if global_config.data_version == "all_turbines":
        trains, vals, tests = [], [], []
        for i in df.id.unique():
            train, val, test = split_one(df.query("id == @i"))
            trains.append(train)
            vals.append(val)
            tests.append(test)
        return pd.concat(trains), pd.concat(vals), pd.concat(tests)
    else:
        return split_one(df)
