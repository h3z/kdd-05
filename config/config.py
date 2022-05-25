PROJECT_NAME = "kdd-05"
ONLINE = False
RANDOM_STATE = 42
DAY = 144

__wandb__ = {
    "project": PROJECT_NAME,
    "entity": "hzzz",
    # mkdir /wandb/PROJECT_NAME
    "dir": f"/wandb/{PROJECT_NAME}",
    "mode": "online" if ONLINE else "offline",
}

conf = {
    "~lr": 1e-4,
    "~batch_size": 32,
    "~epochs": 20,
    "~early_stopping_patience": 10,
    "~optimizer": "adam",
    "~loss": "mse",
    "scheduler": "linear",  # linear, cosine
    "warmup": 0.1,
    "teacher_forcing_ratio": 0.5,
    # data
    "data_version": "full",  # small, full
    "scaler": "all_col",  # each_col, all_col
    "truncate": 0.98,
    "input_size": 10,
    "input_timesteps": DAY,
    "output_timesteps": DAY * 2,
    # model
    "model": "gru",  # gru, seq2seq, attn_seq2seq
    "hidden_size": 48,
    "num_layer": 2,
}


DATA_SPLIT_SIZE = {"train_size": 223 * DAY, "val_size": 16 * DAY, "test_size": 15 * DAY}
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
