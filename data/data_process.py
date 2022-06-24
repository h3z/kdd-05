import datetime

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
            print("xxx", self.mean, self.std)
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

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # print("preprocess, ", df.shape)

        # 记录窗口内的 nan 数量
        isna = df.active_power.isna()
        window_size = global_config.input_timesteps + global_config.output_timesteps
        df["na_count"] = isna.rolling(window_size).sum().shift(-window_size + 1)

        # df = df.fillna(0)
        time_67 = pd.to_datetime("13:10", format="%H:%M").time()
        df.loc[
            df.query("day == 66 or (day ==  67 and time <= @time_67)").index,
            utils.feature_cols,
        ] = 0
        tmp = df.query(f"active_power.isna()")
        while len(tmp) > 0:
            df.loc[tmp.index, utils.feature_cols] = df.loc[
                tmp.index + 1, utils.feature_cols
            ].values
            tmp = df.query(f"active_power.isna()")

        df = self.fe(df)

        df.active_power = df.active_power.clip(lower=0)
        df.reactive_power = df.reactive_power.clip(lower=0)

        if global_config.truncate > 0:
            p = global_config.truncate
            for i in utils.feature_cols:
                lower = df[i].quantile(1 - p)
                upper = df[i].quantile(p)
                df[i] = df[i].clip(lower=lower, upper=upper)

        df[utils.feature_cols] = self.scaler.transform(df[utils.feature_cols].values)
        # df.spd = df.spd * 10

        # print("preprocess done, ", df.shape)

        return df

    def fe(self, df):
        df = self.fe_1_is_vald(df)
        return df

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        t = self.scaler.inverse_transform(preds)
        t = np.clip(t, a_min=0, a_max=t.max())
        return t

    def fe_1_is_vald(self, df):
        df["is_valid"] = df.index.isin(
            df.query(
                "not (active_power <= 0 and spd > 2.5) \
                    or (pab1 > 89 or pab2 > 89 or pab3 > 89) \
                    or (dir < -180 or dir > 180) \
                    or (nacelle_dir < -720 or nacelle_dir > 720)"
            ).index
        ).astype(int)
        return df
