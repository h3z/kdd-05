PROJECT_NAME = "kdd-05"
ONLINE = False
RANDOM_STATE = 42

__wandb__ = {
    "project": PROJECT_NAME,
    "entity": "hzzz",
    # mkdir /wandb/PROJECT_NAME
    "dir": f"/wandb/{PROJECT_NAME}",
    "mode": "online" if ONLINE else "offline",
}

conf = {
    "~lr": 1e-4,
    "~batch_size": 2560,
    "~epochs": 1,
    "~early_stopping_patience": 5,
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
    "input_timesteps": 144,
    "output_timesteps": 144 * 2,
    # model
    "model": "gru",  # gru, seq2seq, attn_seq2seq
    "hidden_size": 48,
    "num_layer": 1,
}
