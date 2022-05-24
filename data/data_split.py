import pandas as pd
from typing import List
from config.config import RANDOM_STATE
from sklearn.model_selection import train_test_split

from config.config import DAY, DATA_SPLIT_SIZE


def split(df: pd.DataFrame) -> List[pd.DataFrame]:

    train_size = DATA_SPLIT_SIZE["train_size"]
    val_size = DATA_SPLIT_SIZE["val_size"]
    test_size = DATA_SPLIT_SIZE["test_size"]

    borders1 = [
        0,
        0 + train_size - DAY,
        0 + train_size + val_size - DAY,
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
