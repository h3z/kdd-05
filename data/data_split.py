from typing import List

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

from config.config import DATA_SPLIT_SIZE, DAY, RANDOM_STATE


def split(df: pd.DataFrame) -> List[pd.DataFrame]:

    train_size = DATA_SPLIT_SIZE["train_size"]
    val_size = DATA_SPLIT_SIZE["val_size"]
    test_size = DATA_SPLIT_SIZE["test_size"]

    borders1 = [
        0,
        0 + train_size - wandb.config.input_timesteps,
        0 + train_size + val_size - wandb.config.input_timesteps,
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
