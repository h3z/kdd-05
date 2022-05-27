import pandas as pd

import utils


class DataReader:
    def __init__(self, turbine_id=None):
        DATA_ROOT = "/home/yanhuize/kdd2022/dataset/sdwpf/"
        self.train = pd.read_pickle(f"{DATA_ROOT}/wtbdata_245days.pkl").rename(
            columns=utils.to_custom_names
        )
        if turbine_id is not None:
            self.train = self.train.query("id == @turbine_id")

        self.location = pd.read_pickle(
            f"{DATA_ROOT}/sdwpf_baidukddcup2022_turb_location.pkl"
        )

    def submit(self, preds):
        pass
