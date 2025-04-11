import torch
import torch.nn as nn

class BodyEncoder(nn.Module):
    def __init__(self, input_size=38, hidden_size=128, num_layers=2, dropout=0.3):
        super(BodyEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x):
        outputs, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        return outputs
