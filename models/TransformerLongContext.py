import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Model(nn.Module):
    """
    Vanilla Transformer with long continuous context window (no retrieval).

    Serves as the continuous-context baseline in the paper. Each position in
    the input sequence attends to every other position, allowing us to measure
    how attention quality degrades as the window grows.
    """

    def __init__(self, configs):
        super().__init__()
        if torch.cuda.is_available() and configs.use_gpu:
            self.device = torch.device(f'cuda:{configs.gpu}')
        elif torch.backends.mps.is_available() and configs.use_gpu:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout

        self.input_projection = nn.Linear(self.channels, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=configs.seq_len + 100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.e_layers
        )

        self.output_projection = nn.Linear(self.d_model, self.channels)
        self.pred_head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, index=None, mode='train'):
        """
        Args:
            x_enc: [batch, seq_len, channels]
        Returns:
            predictions: [batch, pred_len, channels]
        """
        means = x_enc.mean(dim=1, keepdim=True)
        x_enc = x_enc - means

        x = self.input_projection(x_enc)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_projection(x)

        x = x.transpose(1, 2)                # [B, C, seq_len]
        predictions = self.pred_head(x)       # [B, C, pred_len]
        predictions = predictions.transpose(1, 2)  # [B, pred_len, C]

        predictions = predictions + means
        return predictions
