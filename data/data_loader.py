import random
from typing import List

import numpy as np
import pandas as pd
import torch
import wandb

from config.config import feature_cols


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.input_timesteps = wandb.config.input_timesteps
        self.data = data

        input_steps = wandb.config.input_timesteps
        output_steps = wandb.config.output_timesteps
        self.total_timesteps = input_steps + output_steps
        self.len = len(data) - self.total_timesteps + 1

    def __getitem__(self, index):
        x = self.data[index[: self.input_timesteps], :]
        y = self.data[index[self.input_timesteps :], -1:]
        return x, y

    def __len__(self):
        return self.len


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data: np.ndarray, shuffle: bool) -> None:
        super().__init__(data)

        input_steps = wandb.config.input_timesteps
        output_steps = wandb.config.output_timesteps
        self.total_timesteps = input_steps + output_steps
        self.len = len(data) - self.total_timesteps + 1
        self.shuffle = shuffle

    def __iter__(self) -> List[int]:
        lst = list(range(self.len))
        if self.shuffle:
            random.shuffle(lst)
        for i in lst:
            yield list(range(i, i + self.total_timesteps))

    def __len__(self) -> int:
        return self.len


class DataLoader:
    def __init__(self, df: pd.DataFrame, is_train=False) -> None:
        self.is_train = is_train

        df = df[feature_cols]
        # smaller size for training
        if wandb.config.data_version == "small" and is_train:
            self.data = df.values[: len(df) // 30]
        else:
            self.data = df.values

    def get(self) -> torch.utils.data.DataLoader:
        dataset = Dataset(self.data)

        sampler = Sampler(self.data, shuffle=self.is_train)
        batch_size = wandb.config["~batch_size"] if self.is_train else len(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=self.is_train,
        )
