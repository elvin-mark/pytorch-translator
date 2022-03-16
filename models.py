import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, num_words, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_words, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, h):
        o = self.embedding(x).view((1, 1, -1))
        return self.gru(o, h)

    def init_hidden(self):
        return torch.zeros((1, 1, self.hidden_size))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        o = self.embedding(x).view((1, 1, -1))
        o = self.relu(o)
        o, h = self.gru(o, h)
        o = self.logsoftmax(self.out(o[0]))
        return o, h

    def init_hidden(self):
        return torch.zeros((1, 1, self.hidden_size))


class AttentionDecoderRNN(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def init_hidden(self):
        pass
