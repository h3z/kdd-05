import random
from typing import List

import numpy as np
import pandas as pd
import torch

from config.config import global_config
from data.transformer import transformer_data_loader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_df: pd.DataFrame, turbine_count: int, sampler_len: int):
        self.input_timesteps = global_config.input_timesteps

        input_steps = global_config.input_timesteps
        output_steps = global_config.output_timesteps
        self.total_timesteps = input_steps + output_steps
        self.len = sampler_len

        # 为了解决 x 和 y 共用一列数据，并且顺序不能乱的问题
        data_df["__is_valid"] = data_df["is_valid"]
        self.data = (
            torch.tensor(data_df[global_config.features + ["__is_valid"]].values)
            .to(torch.float32)
            .cuda()
        )

    def __getitem__(self, index):
        mid = index + self.input_timesteps
        # : -> :-1，最后一列新增了 is_valid
        x = self.data[index:mid, :-1]
        # 这个 -1 要注意。可能随着特征变化，就不再指向 y 了。
        # 目前主要是自定义损失时候，想要通过 y 传递一些权重过去
        # 2022年06月21日16:57:05 -1 -> -2，新增了 is_valid
        y = self.data[mid : index + self.total_timesteps, -2:-1]
        w = self.data[mid : index + self.total_timesteps, -1]
        return x, y, w

    def __len__(self):
        return self.len


class TransformerDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, turbine_count: int, sampler_len: int):
        super().__init__(data_df, turbine_count, sampler_len)

    def __getitem__(self, index):
        window = self.data_df[index : index + self.total_timesteps]
        src, trg, trg_y = transformer_data_loader.get_src_trg(window)

        trg = trg.clone()
        trg[1:] = 0
        # if not self.is_train:
        #     trg = trg.clone()
        #     trg[1:] = 0
        # trg = trg[:, -1]

        # TODO loss这里要修
        trg_y = trg_y[:, -1]

        return (src, trg), trg_y


class Sampler(torch.utils.data.Sampler):
    def __init__(
        self, data_df: pd.DataFrame, turbine_count: int, is_train: bool
    ) -> None:
        super().__init__(data_df)
        self.is_train = is_train

        if global_config.distributed:
            self.rank = global_config.cuda_rank
            self.num_replicas = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        total_timesteps = global_config.input_timesteps + global_config.output_timesteps
        data_len = len(data_df)

        if self.is_train:
            lst = torch.randperm(data_len).tolist()
        else:
            lst = list(range(data_len))
        each_turbine = data_len / turbine_count
        threshold = each_turbine - (total_timesteps - 1)
        lst = [i for i in lst if i % each_turbine < threshold]
        lst = np.array(lst)[data_df.iloc[lst].na_count < 6]

        self.len = len(lst)
        self.lst = lst

    def __iter__(self) -> List[int]:
        return iter(self.distributed(self.lst) if self.is_train else self.lst)

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

        self.data_df = df

    def get(self) -> torch.utils.data.DataLoader:
        sampler = Sampler(self.data_df, self.turbine_count, is_train=self.is_train)

        if global_config.model == "transformer":
            dataset = TransformerDataset(self.data_df, self.turbine_count, len(sampler))
        else:
            dataset = Dataset(self.data_df, self.turbine_count, len(sampler))

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=global_config["~batch_size"],
            drop_last=self.is_train,
            num_workers=0,
            # pin_memory=True,
            # persistent_workers=True,
        )
