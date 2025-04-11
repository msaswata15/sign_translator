import torch
import torch.nn as nn

class HandEncoder(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, num_layers=2, dropout=0.3):
        super(HandEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        outputs, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        return outputs
