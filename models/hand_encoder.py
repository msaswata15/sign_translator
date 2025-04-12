import torch.nn as nn

class HandEncoder(nn.Module):
    def __init__(self, input_size=84, hidden_size=256, num_layers=2, dropout=0.3):
        super(HandEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        outputs, _ = self.lstm(x)  # outputs: (batch, seq_len, 2*hidden_size)
        return outputs