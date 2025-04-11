import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, hand_encoder, face_encoder, body_encoder, decoder, fusion, device):
        super().__init__()
        self.hand_encoder = hand_encoder
        self.face_encoder = face_encoder
        self.body_encoder = body_encoder
        self.decoder = decoder
        self.fusion = fusion
        self.device = device

    def forward(self, hand, face, body, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode each modality
        hand_out = self.hand_encoder(hand)
        face_out = self.face_encoder(face)
        body_out = self.body_encoder(body)

        # Fuse them by concatenation and apply attention
        combined = torch.cat((hand_out, face_out, body_out), dim=2)
        context, _ = self.fusion(combined)

        hidden = torch.zeros(1, batch_size, self.decoder.lstm.hidden_size).to(self.device)
        cell = torch.zeros(1, batch_size, self.decoder.lstm.hidden_size).to(self.device)

        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, context, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
