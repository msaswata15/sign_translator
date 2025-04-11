import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch, seq_len, hidden_size)
        attn_scores = self.attn(encoder_outputs)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(encoder_outputs * attn_weights, dim=1)  # (batch, hidden_size)
        return context, attn_weights
