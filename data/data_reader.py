import pandas as pd

from config import config as C


class DataReader:
    def __init__(self):
        DATA_ROOT = "/home/yanhuize/kdd2022/dataset/sdwpf/"
        self.train = pd.read_pickle(f"{DATA_ROOT}/wtbdata_245days.pkl").rename(
            columns=C.to_custom_names
        )
        self.location = pd.read_pickle(
            f"{DATA_ROOT}/sdwpf_baidukddcup2022_turb_location.pkl"
        )


    def submit(self, preds):
        pass
