import torch
import torch.nn as nn
import math

class DQNAgentLSTM(nn.Module):
    """DQN Agent with an LSTM backend."""
    model_type = "lstm" 
    def __init__(self, input_size: int, hidden_size: int, num_actions: int):
        super(DQNAgentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

        # Initialize forget gate bias to 1.0, a common practice for LSTMs
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Clamp lengths to be at least 1 to avoid errors with empty sequences
        lengths_cpu = torch.clamp(lengths, min=1).cpu()
        
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last valid output for each sequence
        last_outputs = output[torch.arange(len(output)), lengths_cpu - 1]
        return self.fc(last_outputs)

class PositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer models."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2) # (seq_len, batch, feature)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.permute(1, 0, 2)) # (batch, seq_len, feature)

class DQNAgentTransformer(nn.Module):
    """DQN Agent with a Transformer Encoder backend."""
    model_type = "transformer" 
    def __init__(self, input_size: int, hidden_size: int, num_actions: int, n_heads: int, num_layers: int, dropout: float):
        super(DQNAgentTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_actions)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) * math.sqrt(self.hidden_size)
        x = self.positional_encoding(x)

        # Create a mask for padded elements
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Get the last valid output for each sequence
        last_outputs = output[torch.arange(len(output)), lengths.cpu() - 1]
        return self.fc(last_outputs)