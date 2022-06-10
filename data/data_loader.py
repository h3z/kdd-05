import random
from typing import List

import numpy as np
import pandas as pd
import torch

from config.config import global_config
from utils import feature_cols


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.input_timesteps = global_config.input_timesteps
        self.data = data

        input_steps = global_config.input_timesteps
        output_steps = global_config.output_timesteps
        self.total_timesteps = input_steps + output_steps
        if global_config.data_version == "all_turbines":
            self.len = len(data) - (self.total_timesteps - 1) * 134
        else:
            self.len = len(data) - (self.total_timesteps - 1)

    def __getitem__(self, index):
        mid = index + self.input_timesteps
        x = self.data[index:mid, :]
        y = self.data[mid : index + self.total_timesteps, -1:]

        return x, y

    def __len__(self):
        return self.len


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data: np.ndarray, is_train: bool) -> None:
        super().__init__(data)
        self.is_train = is_train

        if global_config.distributed:
            self.rank = global_config.cuda_rank
            self.num_replicas = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        self.total_timesteps = (
            global_config.input_timesteps + global_config.output_timesteps
        )
        self.total_len = len(data) - self.total_timesteps + 1
        self.len = self.total_len // self.num_replicas

    def __iter__(self) -> List[int]:
        if self.is_train:
            lst = torch.randperm(self.total_len).tolist()
        else:
            lst = list(range(self.total_len))

        return iter(self.distributed(lst) if self.is_train else lst)

    def distributed(self, lst):
        return lst[self.rank : len(lst) : self.num_replicas]
        # return lst[self.rank : len(lst) : self.num_replicas]

    def __len__(self) -> int:
        return self.len


class AllTurbinesSampler(Sampler):
    def __init__(self, data: np.ndarray, is_train: bool) -> None:
        super().__init__(data, is_train)

        total_len = len(data) - (self.total_timesteps - 1) * 134
        self.len = total_len // self.num_replicas
        self.data_len = len(data)

    def __iter__(self) -> List[int]:
        if self.is_train:
            lst = torch.randperm(self.data_len).tolist()
        else:
            lst = list(range(self.data_len))
        each_turbine = self.data_len / 134
        threshold = each_turbine - (self.total_timesteps - 1)
        lst = [i for i in lst if i % each_turbine < threshold]
        return iter(self.distributed(lst) if self.is_train else lst)


class DataLoader:
    def __init__(self, df: pd.DataFrame, is_train=False) -> None:
        self.is_train = is_train

        df = df[feature_cols]
        # smaller size for training
        if global_config.data_version == "small" and is_train:
            self.data = df.values[: len(df) // 30]
        else:
            self.data = df.values
        self.data = torch.tensor(self.data).to(torch.float32).cuda()

    def get(self) -> torch.utils.data.DataLoader:
        dataset = Dataset(self.data)

        if global_config.data_version == "all_turbines":
            sampler = AllTurbinesSampler(self.data, is_train=self.is_train)
        else:
            sampler = Sampler(self.data, is_train=self.is_train)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=global_config["~batch_size"],
            drop_last=self.is_train,
            num_workers=0,
            # pin_memory=True,
            # persistent_workers=True,
        )
