import pandas as pd

import utils


class DataReader:
    def __init__(self, turbine_id=None, col=None):
        DATA_ROOT = "/home/yanhuize/kdd2022/dataset/sdwpf/"
        self.train = pd.read_pickle(f"{DATA_ROOT}/wtbdata_245days.pkl").rename(
            columns=utils.to_custom_names
        )

        self.location = pd.read_pickle(
            f"{DATA_ROOT}/sdwpf_baidukddcup2022_turb_location.pkl"
        )
        self.location = self.location_col(self.location)

        if turbine_id is not None:
            self.train = self.train.query("id == @turbine_id")
        elif col is not None and col != -1:
            # 选择第 N 列
            ids = self.location.query("col == @col").TurbID.values
            self.train = self.train.query("id in @ids")

    def location_col(self, location):
        location["col"] = -1
        for i, v in enumerate([500, 1500, 2500, 3800, 4800, 10000]):
            location.loc[(location.col == -1) & (location.x < v), "col"] = i
        return location

    def submit(self, preds):
        pass
