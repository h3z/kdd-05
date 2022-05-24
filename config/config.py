PROJECT_NAME = "kdd-05"
ONLINE = False

__wandb__ = {
    "project": PROJECT_NAME,
    "entity": "hzzz",
    # mkdir /wandb/PROJECT_NAME
    "dir": f"/wandb/{PROJECT_NAME}",
    "mode": "online" if ONLINE else "offline",
}


RANDOM_STATE = 42

DAY = 144
DATA_SPLIT_SIZE = {"train_size": 153 * DAY, "val_size": 16 * DAY, "test_size": 15 * DAY}

to_custom_names = {
    "TurbID": "id",
    "Day": "day",
    "Tmstamp": "time",
    "Wspd": "spd",
    "Wdir": "dir",
    "Etmp": "environment_tmp",
    "Itmp": "inside_tmp",
    "Ndir": "nacelle_dir",
    "Pab1": "pab1",
    "Pab2": "pab2",
    "Pab3": "pab3",
    "Prtv": "reactive_power",
    "Patv": "active_power",
}

to_origin_names = {
    "id": "TurbID",
    "day": "Day",
    "time": "Tmstamp",
    "spd": "Wspd",
    "dir": "Wdir",
    "environment_tmp": "Etmp",
    "inside_tmp": "Itmp",
    "nacelle_dir": "Ndir",
    "pab1": "Pab1",
    "pab2": "Pab2",
    "pab3": "Pab3",
    "reactive_power": "Prtv",
    "active_power": "Patv",
}

feature_cols = [
    "spd",
    "dir",
    "environment_tmp",
    "inside_tmp",
    "nacelle_dir",
    "pab1",
    "pab2",
    "pab3",
    "reactive_power",
    "active_power",
]
