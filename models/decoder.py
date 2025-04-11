import torch
import torch.nn as nn

class NLPDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout):
        super(NLPDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim + enc_hidden_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, context, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell
