import torch
import torch.nn as nn
from torch import Tensor, nn

from config.config import global_config as config
from model.base_model import BaseModelApp
from model.transformer import positional_encoder as pe


def generate_square_subsequent_mask(dim1: int, dim2: int, dim3: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, batch_size * n_heads
        dim2: int. For src and trg masking this must be target sequence length.
        dim3: int. For src masking, this must be encoder sequence length.
              For trg masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2, dim3]
    """
    return torch.triu(torch.ones(dim1, dim2, dim3) * float("-inf"), diagonal=1)


class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".
    A detailed description of the code can be found in my article here:
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.
    Unlike the paper, this class assumes that input layers, positional encoding
    layers and linear mapping layers are separate from the encoder and decoder,
    i.e. the encoder and decoder only do what is depicted as their sub-layers
    in the paper. For practical purposes, this assumption does not make a
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the
    Encoder() and Decoder() classes.
    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020).
    'Deep Transformer Models for Time Series Forecasting:
    The Influenza Prevalence Case'.
    arXiv:2001.08317 [cs, stat] [Preprint].
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).
    [2] Vaswani, A. et al. (2017)
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint].
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    """

    def __init__(
        self,
        input_size: int,
        dec_seq_len: int,
        max_seq_len: int,
        out_seq_len: int = 58,
        dim_val: int = 512,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        dropout_encoder: float = 0.2,
        dropout_decoder: float = 0.2,
        dropout_pos_enc: float = 0.2,
        dim_feedforward_encoder: int = 2048,
        dim_feedforward_decoder: int = 2048,
    ):

        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            max_seq_len: int, length of the longest sequence the model will
                         receive. Used in positional encoding.
            out_seq_len: int, the length of the model's output (i.e. the target
                         sequence length)
            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len

        print("input_size is: {}".format(input_size))
        print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=input_size, out_features=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=out_seq_len * dim_val, out_features=out_seq_len
        )

        # Create positional encoder
        self.positional_encoding_layer = pe.PositionalEncoder(
            d_model=dim_val, dropout=dropout_pos_enc
        )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=True,
        )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=True,
        )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None
        )

        # Input length
        enc_seq_len = max_seq_len

        # Output length
        output_sequence_length = out_seq_len

        # Heads in attention layers
        n_heads = n_heads

        batch_size = config["~batch_size"]

        # Make src mask for decoder with size:
        # [batch_size*n_heads, output_sequence_length, enc_seq_len]
        """
        src_mask (batch, 288, 144)
        src_mask[0] 288,144
        src_mask[0][0] == [0, -inf, -inf,... -inf] 预测第一个值的时候，只能用第一个数，即过去一刻
        src_mask[0][1] == [0, 0,    -inf,... -inf] 预测第二个值的时候，只能用前两个数，即过去两刻

        tgt_mask (batch, 288, 288)
        同 src_mask, 只不过 -inf 有两百多个

        是作用在 decoder's input 上的。
        """
        self.src_mask = generate_square_subsequent_mask(
            dim1=batch_size * n_heads, dim2=output_sequence_length, dim3=enc_seq_len
        ).cuda()

        # Make tgt mask for decoder with size:
        # [batch_size*n_heads, output_sequence_length, output_sequence_length]
        self.tgt_mask = generate_square_subsequent_mask(
            dim1=batch_size * n_heads,
            dim2=output_sequence_length,
            dim3=output_sequence_length,
        ).cuda()

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
    ) -> Tensor:
        """
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the feature number
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, E is the feature number.
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src)

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src)

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder(src=src)

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt)

        src_mask, tgt_mask = self.get_mask(batch_size=len(src))
        # Pass throguh decoder
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )

        # Pass through the linear mapping layer
        decoder_output = self.linear_mapping(decoder_output.flatten(start_dim=1))

        return decoder_output

    def get_mask(self, batch_size):
        if batch_size == config["~batch_size"]:
            return self.src_mask, self.tgt_mask

        src_mask = generate_square_subsequent_mask(
            dim1=batch_size * config.n_heads,
            dim2=config.output_timesteps,
            dim3=config.input_timesteps,
        ).cuda()

        # Make tgt mask for decoder with size:
        # [batch_size*n_heads, output_sequence_length, output_sequence_length]
        tgt_mask = generate_square_subsequent_mask(
            dim1=batch_size * config.n_heads,
            dim2=config.output_timesteps,
            dim3=config.output_timesteps,
        ).cuda()

        return src_mask, tgt_mask


def get_model():
    ## Model parameters
    dim_val = (
        config.dim_val
    )  # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = (
        config.n_heads
    )  # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4  # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4  # Number of times the encoder layer is stacked in the encoder
    # TODO maybe it is len of features ?
    input_size = 10  # The number of input variables. 1 if univariate forecasting.
    enc_seq_len = (
        config.input_timesteps
    )  # length of input given to encoder. Can have any integer value.
    # TODO maybe it should be 1?  I dont know what means decoder's input.
    dec_seq_len = (
        config.output_timesteps
    )  # length of input given to decoder. Can have any integer value.

    output_sequence_length = (
        config.output_timesteps
    )  # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len  # What's the longest sequence the model will encounter? Used to make the positional encoder

    model = TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size,
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length,
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
    )

    return model


class ModelApp(BaseModelApp):
    def __init__(self, *models) -> None:
        super().__init__(*models)

    def forward(self, batch_x, batch_y, is_training=False):
        return self.model(*batch_x)
