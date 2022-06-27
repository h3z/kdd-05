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

        # 到这里，原始数据 -> 加了 na_count -> 处理 na
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

        df = self.fe_pab(df)
        df = self.fe_spd(df)
        df = self.fe_dir(df)
        df = self.fe_day(df)
        df = self.fe_nacelle_dir(df)
        df = self.fe_active_power(df)
        return df

    def fe_day(self, df):
        df["week"] = df.day % 7
        # df["month"] = df.day // 30
        df["hour"] = df.time.apply(lambda x: str(x).split(":")[0]).astype(int)
        return df

    def fe_time(self, df):
        pass

    def fe_spd(self, df):
        dir = df[["pab1", "pab2", "pab3"]].mean(1)
        df["spd_pab"] = df.spd * np.cos(np.deg2rad(dir))

        df.spd_pab = (df.spd_pab - 4.282714659291982) / 3.6770292968518103

        diff = df.spd - df.spd.shift(-1)
        diff = diff.fillna(0)
        # df.spd_diff = (df.spd_diff - 5.028376377350847) / 3.3937034012387226
        df["spd_diff"] = diff / 17

        return df

    def fe_dir(self, df):
        df["dir_cos"] = np.cos(np.deg2rad(df.dir))
        df["dir_sin"] = np.sin(np.deg2rad(df.dir))
        df["dir_"] = df.dir / 90
        df["dir_abs"] = abs(df.dir_)

        return df

    def fe_environment_tmp(self, df):
        pass

    def fe_nacelle_dir(self, df):
        df["nacelle_dir_cos"] = np.cos(np.deg2rad(df.nacelle_dir))
        df["nacelle_dir_sin"] = np.sin(np.deg2rad(df.nacelle_dir))
        df["nacelle_dir_"] = (df.nacelle_dir + 720) % 360 / 360

        return df

    def fe_pab(self, df):
        df["pab_mean"] = df[["pab1", "pab2", "pab3"]].mean(1)
        df["pab_mean"] /= 90
        df["pab_std"] = df[["pab1", "pab2", "pab3"]].std(1)
        df["pab_cv"] = df.pab_std / (df.pab_mean + 1e-10)

        df["pab1_"] = df.pab1 / 90
        df["pab2_"] = df.pab2 / 90
        df["pab3_"] = df.pab3 / 90

        df["pab1_cos"] = np.cos(np.deg2rad(df.pab1))
        df["pab2_cos"] = np.cos(np.deg2rad(df.pab2))
        df["pab3_cos"] = np.cos(np.deg2rad(df.pab3))

        df["pab1_is20"] = (df.pab1 > 20).astype(int)
        df["pab2_is20"] = (df.pab2 > 20).astype(int)
        df["pab3_is20"] = (df.pab3 > 20).astype(int)

        return df

    def fe_active_power(self, df):
        diff = df.active_power - df.active_power.shift(-1)
        diff = diff.fillna(0)
        # df["active_power_diff"] = (diff - 350.4457511390552) / 424.9932085867589
        df["active_power_diff"] = diff / 1500
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
