import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_scores = self.attn(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * x, dim=1)  # (batch, input_dim)
        return context, attn_weights

class ModalityFusion(nn.Module):
    def __init__(self, hand_dim, face_dim, body_dim, fused_dim):
        super().__init__()
        self.proj_hand = nn.Linear(hand_dim, fused_dim)
        self.proj_face = nn.Linear(face_dim, fused_dim)
        self.proj_body = nn.Linear(body_dim, fused_dim)
        self.attn = SpatioTemporalAttention(fused_dim)
    
    def forward(self, hand_feat, face_feat, body_feat):
        # Each input: (batch, seq_len, dim)
        hand_proj = self.proj_hand(hand_feat)
        face_proj = self.proj_face(face_feat)
        body_proj = self.proj_body(body_feat)
        
        # Combine modalities by addition
        fused = hand_proj + face_proj + body_proj
        context, attn_weights = self.attn(fused)
        return context, fused, attn_weights