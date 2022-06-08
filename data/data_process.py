import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import utils
from config.config import global_config


class Scaler:
    def __init__(self) -> None:
        return

    def fit(self, data):
        if global_config.scaler == "each_col":
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        elif global_config.scaler == "all_col":
            self.mean = np.mean(data)
            self.std = np.std(data)
        elif global_config.scaler == "constant":
            # 方便打分时，预处理值固定
            self.mean = global_config.__mean__
            self.std = global_config.__std__
        elif global_config.scaler == "all_turbines_all_col":
            # 方便打分时，预处理值固定
            self.mean = 68.12220790619038
            self.std = 186.9209197502287

        # print(f"mean: {self.mean}", f"std: {self.std}")

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if global_config.scaler == "each_col":
            return (data * self.std[-1]) + self.mean[-1]
        elif global_config.scaler in ["all_col", "constant", "all_turbines_all_col"]:
            return (data * self.std) + self.mean


class DataProcess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.scaler = Scaler()
        df = df.fillna(0)

        self.scaler.fit(df[utils.feature_cols].values)

        # self.scaler = StandardScaler()
        # self.numerical_cols = [...]
        # self.scaler.fit(df[self.numerical_cols].fillna(0).values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # print("preprocess, ", df.shape)
        df = df.fillna(0)
        df.active_power = df.active_power.clip(lower=0)
        df.reactive_power = df.reactive_power.clip(lower=0)

        if global_config.truncate > 0:
            p = global_config.truncate
            for i in utils.feature_cols:
                lower = df[i].quantile(1 - p)
                upper = df[i].quantile(p)
                df[i] = df[i].clip(lower=lower, upper=upper)

        df[utils.feature_cols] = self.scaler.transform(df[utils.feature_cols].values)
        # print("preprocess done, ", df.shape)
        return df

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(preds)
