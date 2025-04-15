import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Seq2Seq(nn.Module):
    def __init__(self, hand_encoder, face_encoder, body_encoder, fusion, decoder, device):
        super().__init__()
        self.hand_encoder = hand_encoder
        self.face_encoder = face_encoder
        self.body_encoder = body_encoder
        self.fusion = fusion
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, keypoint_lengths, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        # Split src into modalities
        hand_seq = src[:, :, :84]
        face_seq = src[:, :, 84:224]
        body_seq = src[:, :, 224:299]

        # Ensure lengths are on CPU as required by pack_padded_sequence
        keypoint_lengths = torch.tensor(keypoint_lengths, dtype=torch.long).cpu()

        # Pack sequences
        hand_packed = pack_padded_sequence(hand_seq, keypoint_lengths, batch_first=True, enforce_sorted=False)
        face_packed = pack_padded_sequence(face_seq, keypoint_lengths, batch_first=True, enforce_sorted=False)
        body_packed = pack_padded_sequence(body_seq, keypoint_lengths, batch_first=True, enforce_sorted=False)

        # Encode
        hand_out_packed, _ = self.hand_encoder.lstm(hand_packed)
        face_out_packed, _ = self.face_encoder.lstm(face_packed)
        body_out_packed, _ = self.body_encoder.lstm(body_packed)

        # Unpack sequences
        hand_out, _ = pad_packed_sequence(hand_out_packed, batch_first=True)
        face_out, _ = pad_packed_sequence(face_out_packed, batch_first=True)
        body_out, _ = pad_packed_sequence(body_out_packed, batch_first=True)

        context, fused, _ = self.fusion(hand_out, face_out, body_out)

        hidden = src.new_zeros(1, batch_size, self.decoder.lstm.hidden_size).float().to(self.device)
        cell = src.new_zeros(1, batch_size, self.decoder.lstm.hidden_size).float().to(self.device)

        input_token = trg[:, 0]  # <sos>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, context, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        
        return outputs

    def translate(self, src, max_len, keypoint_lengths=None):
        batch_size = src.size(0)
        
        # Split src into modalities
        hand_seq = src[:, :, :84]
        face_seq = src[:, :, 84:224]
        body_seq = src[:, :, 224:299]

        with torch.no_grad():
            if keypoint_lengths is not None:
                keypoint_lengths = torch.tensor(keypoint_lengths, dtype=torch.long).cpu()
                hand_packed = pack_padded_sequence(hand_seq, keypoint_lengths, batch_first=True, enforce_sorted=False)
                face_packed = pack_padded_sequence(face_seq, keypoint_lengths, batch_first=True, enforce_sorted=False)
                body_packed = pack_padded_sequence(body_seq, keypoint_lengths, batch_first=True, enforce_sorted=False)

                hand_out_packed, _ = self.hand_encoder.lstm(hand_packed)
                face_out_packed, _ = self.face_encoder.lstm(face_packed)
                body_out_packed, _ = self.body_encoder.lstm(body_packed)

                hand_out, _ = pad_packed_sequence(hand_out_packed, batch_first=True)
                face_out, _ = pad_packed_sequence(face_out_packed, batch_first=True)
                body_out, _ = pad_packed_sequence(body_out_packed, batch_first=True)
            else:
                hand_out = self.hand_encoder(hand_seq)
                face_out = self.face_encoder(face_seq)
                body_out = self.body_encoder(body_seq)

            context, _, _ = self.fusion(hand_out, face_out, body_out)

            hidden = src.new_zeros(1, batch_size, self.decoder.lstm.hidden_size).float().to(self.device)
            cell = src.new_zeros(1, batch_size, self.decoder.lstm.hidden_size).float().to(self.device)

            outputs = []
            input_token = torch.tensor([1] * batch_size).to(self.device)  # Assuming <sos> token id is 1
            for _ in range(max_len):
                output, hidden, cell = self.decoder(input_token, context, hidden, cell)
                top1 = output.argmax(1)
                outputs.append(top1.unsqueeze(1))
                input_token = top1
            outputs = torch.cat(outputs, dim=1)

        return outputs
