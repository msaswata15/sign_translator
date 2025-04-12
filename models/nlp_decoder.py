import torch
import torch.nn as nn

class NLPDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_dim, hidden_dim, dropout):
        super(NLPDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim + context_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, context, hidden, cell):
        # input: (batch,) token indices
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input))  # (batch, 1, emb_dim)
        # Concatenate context: (batch, context_dim)
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # (batch, vocab_size)
        return prediction, hidden, cell