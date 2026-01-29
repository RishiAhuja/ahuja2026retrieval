import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Model(nn.Module):
    """
    Time-CAG: Transformer with Long Context (No Retrieval)
    Testing hypothesis: Long continuous context > Retrieved short contexts
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Device selection (MPS for Mac, CUDA for GPU, CPU as fallback)
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
        
        # Model architecture parameters
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        
        # Input projection: map each channel to d_model dimensions
        self.input_projection = nn.Linear(self.channels, self.d_model)
        
        # Positional encoding for sequence position information
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=configs.seq_len + 100)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=self.e_layers
        )
        
        # Output projection: map from d_model back to prediction
        self.output_projection = nn.Linear(self.d_model, self.channels)
        
        # Prediction head: project sequence length to prediction length
        self.pred_head = nn.Linear(self.seq_len, self.pred_len)
        
        print(f"[Time-CAG] Initialized with seq_len={self.seq_len}, d_model={self.d_model}, "
              f"n_heads={self.n_heads}, layers={self.e_layers}")
        print(f"[Time-CAG] Using device: {self.device}")
        print(f"[Time-CAG] Context Window: {self.seq_len} steps (vs RAFT's default 720)")
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                mask=None, index=None, mode='train'):
        """
        Args:
            x_enc: [batch_size, seq_len, channels] - input time series
            Other args: compatibility with RAFT interface, not used
        Returns:
            predictions: [batch_size, pred_len, channels]
        """
        batch_size, seq_len, channels = x_enc.shape
        
        # Normalize: remove mean from the sequence
        means = x_enc.mean(dim=1, keepdim=True)
        x_enc = x_enc - means
        
        # Project input to d_model dimensions
        # [batch_size, seq_len, channels] -> [batch_size, seq_len, d_model]
        x = self.input_projection(x_enc)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        # This is where the "long context" magic happens
        # Each position can attend to all other positions in the sequence
        x = self.transformer_encoder(x)
        
        # Project back to channel dimensions
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, channels]
        x = self.output_projection(x)
        
        # Transpose for temporal projection
        # [batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Project from seq_len to pred_len
        # [batch_size, channels, seq_len] -> [batch_size, channels, pred_len]
        predictions = self.pred_head(x)
        
        # Transpose back
        # [batch_size, channels, pred_len] -> [batch_size, pred_len, channels]
        predictions = predictions.transpose(1, 2)
        
        # Add back the mean
        predictions = predictions + means
        
        return predictions


class TimeCAG(nn.Module):
    """
    Alias for compatibility - same as TransformerLongContext
    """
    def __init__(self, configs):
        super(TimeCAG, self).__init__()
        self.model = Model(configs)
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
