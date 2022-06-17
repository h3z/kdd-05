import random
from typing import List

import numpy as np
import pandas as pd
import torch

from config.config import global_config
from data.transformer import transformer_data_loader
from utils import feature_cols


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, turbine_count: int):
        self.input_timesteps = global_config.input_timesteps
        self.data = data

        input_steps = global_config.input_timesteps
        output_steps = global_config.output_timesteps
        self.total_timesteps = input_steps + output_steps
        self.len = len(data) - (self.total_timesteps - 1) * turbine_count

    def __getitem__(self, index):
        mid = index + self.input_timesteps
        x = self.data[index:mid, :]
        # 这个 -1 要注意。可能随着特征变化，就不再指向 y 了。
        # 目前主要是自定义损失时候，想要通过 y 传递一些权重过去
        y = self.data[mid : index + self.total_timesteps, -1:]

        return x, y

    def __len__(self):
        return self.len


class TransformerDataset(Dataset):
    def __init__(self, data: np.ndarray, turbine_count: int):
        super().__init__(data, turbine_count)

    def __getitem__(self, index):
        window = self.data[index : index + self.total_timesteps]
        src, trg, trg_y = transformer_data_loader.get_src_trg(window)

        trg = trg.clone()
        trg[1:] = 0
        # if not self.is_train:
        #     trg = trg.clone()
        #     trg[1:] = 0
        # trg = trg[:, -1]

        trg_y = trg_y[:, -1]

        return (src, trg), trg_y


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data: np.ndarray, turbine_count: int, is_train: bool) -> None:
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
        self.total_len = len(data) - (self.total_timesteps - 1) * turbine_count
        # self.total_len //= 10
        self.len = self.total_len // self.num_replicas
        self.turbine_count = turbine_count
        self.data_len = len(data)

    def __iter__(self) -> List[int]:
        if self.is_train:
            lst = torch.randperm(self.data_len).tolist()
        else:
            lst = list(range(self.data_len))
        each_turbine = self.data_len / self.turbine_count
        threshold = each_turbine - (self.total_timesteps - 1)
        lst = [i for i in lst if i % each_turbine < threshold]
        return iter(self.distributed(lst) if self.is_train else lst)

    def distributed(self, lst):
        return lst[self.rank : len(lst) : self.num_replicas]
        # return lst[self.rank : len(lst) : self.num_replicas]

    def __len__(self) -> int:
        return self.len


class DataLoader:
    def __init__(self, df: pd.DataFrame, is_train=False) -> None:
        self.is_train = is_train

        # 过滤后一共多少台机组，滑窗时候需要从这个机组跳到下个机组要用到
        self.turbine_count = df.id.nunique()

        # 特征列，如果有自定义特征，需要在这里加，这里硬编码了，暂时不优化，后期应该会整合到配置文件中。
        df = df[feature_cols]

        # smaller size for training
        if global_config.data_version == "small" and is_train:
            self.data = df.values[: len(df) // 30]
        else:
            self.data = df.values

        self.data = torch.tensor(self.data).to(torch.float32).cuda()

    def get(self) -> torch.utils.data.DataLoader:
        if global_config.model == "transformer":
            dataset = TransformerDataset(self.data, self.turbine_count)
        else:
            dataset = Dataset(self.data, self.turbine_count)

        sampler = Sampler(self.data, self.turbine_count, is_train=self.is_train)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=global_config["~batch_size"],
            drop_last=self.is_train,
            num_workers=0,
            # pin_memory=True,
            # persistent_workers=True,
        )
