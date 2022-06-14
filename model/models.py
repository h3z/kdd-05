from config.config import global_config
from model import attn_seq2seq, base_model, gru, seq2seq
from model.transformer import time_series_transformer


def get(batch_count=None) -> base_model.BaseModelApp:
    if global_config.model == "gru":
        model = gru.SimpleGRU().cuda()
        modelapp = base_model.BaseModelApp(model)
    # summary(model, (global_config.batch_size, 144, 10))

    elif global_config.model == "seq2seq":
        modelapp = seq2seq.ModelApp(
            seq2seq.EncoderRNN().cuda(), seq2seq.DecoderRNN().cuda()
        )

    elif global_config.model == "attn_seq2seq":
        modelapp = attn_seq2seq.ModelApp(
            attn_seq2seq.EncoderRNN().cuda(),
            attn_seq2seq.AttnDecoderRNN().cuda(),
        )
    elif global_config.model == "transformer":
        modelapp = time_series_transformer.ModelApp(
            time_series_transformer.get_model().cuda()
        )

    if batch_count is not None:
        modelapp.init_scheduler(batch_count)

    return modelapp
