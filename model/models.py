import wandb

from model import attn_seq2seq, base_model, gru, seq2seq


def get(batch_count=None) -> base_model.BaseModelApp:
    if wandb.config.model == "gru":
        model = gru.SimpleGRU().to("cuda")
        modelapp = base_model.BaseModelApp(model)
    # summary(model, (wandb.config.batch_size, 144, 10))

    elif wandb.config.model == "seq2seq":
        modelapp = seq2seq.ModelApp(
            seq2seq.EncoderRNN().to("cuda"), seq2seq.DecoderRNN().to("cuda")
        )

    elif wandb.config.model == "attn_seq2seq":
        modelapp = attn_seq2seq.ModelApp(
            attn_seq2seq.EncoderRNN().to("cuda"),
            attn_seq2seq.AttnDecoderRNN().to("cuda"),
        )

    if batch_count is not None:
        modelapp.init_scheduler(batch_count)

    return modelapp
