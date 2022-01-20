import torch
import torch.nn as nn


class SimpleRnn(nn.Module):
    """Simple Rnn Cell"""

    def __init__(self, input_size, hidden_size, batch_size):
        super(SimpleRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnn_cell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


class SimpleRnn2(nn.Module):
    """Simpole Rnn Net"""

    def __init__(self, input_size, hidden_size, batch_size, num_class, embedding_size=10, num_layers=1):
        super(SimpleRnn2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_class = num_class
        # embedding 层
        self.emb = nn.Embedding(input_size, embedding_dim=embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 隐藏层做全连接
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # batch first
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # batchSize, seqLen, embeddingSize
        x = self.emb(x)
        x, _ = self.rnn(x, h0)
        # batchSize,seqLen,hiddenSize
        x = self.fc(x)
        # 为了i算交叉熵，调整结果向量的维度
        # batchSize,seqLen,numClass
        return x.view(-1, self.num_class)


class RnnClassifier(nn.Module):
    """Use Rnn for classification"""

    def __init__(self, input_size, hidden_size, num_class, num_layers, embedding_size=10, bidirectional=True):
        super(RnnClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.n_directions = 2 if bidirectional else 1

        self.emb = nn.Embedding(input_size, embedding_dim=embedding_size)
        # self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        # use GRU instead
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * self.n_directions, num_class)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * self.n_directions, batch_size, self.hidden_size).to('cuda')

    def forward(self, x, seq_length):
        batch_size = x.size(0)

        h0 = self.init_hidden(batch_size)
        x = self.emb(x)
        # pack them up
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=seq_length, batch_first=True)

        output, hidden = self.gru(x, h0)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        y = self.fc(hidden_cat)
        return y


class IMDBReview(nn.Module):
    embedding_dim = 200
    hidden_size = 128
    num_layers = 2
    use_cell = True

    def __init__(self, words_size, seq_len, batch_size, classes, bidirection=True, dropout=0.2):
        super(IMDBReview, self).__init__()

        self.seq_len = seq_len
        self.bidirection = bidirection
        self.dropout = dropout
        self.batch_size = batch_size
        self.bi_num = 2 if self.bidirection else 1

        self.emb = nn.Embedding(words_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirection, dropout=self.dropout, batch_first=True)
        self.fc_seq = nn.Linear(self.seq_len * self.hidden_size * self.bi_num, 20)
        self.fc_cell = nn.Linear(self.bi_num * self.hidden_size, 20)

        self.fc = nn.Linear(20, classes)
        self.act1 = nn.LogSoftmax(dim=1)
        self.act2 = nn.LogSoftmax(dim=1)

    def init_hidden_cell(self):
        return (
            torch.normal(0, 1, size=(self.num_layers * self.bi_num, self.batch_size, self.hidden_size)).to('cuda:0'),
            torch.normal(0, 1, size=(self.num_layers * self.bi_num, self.batch_size, self.hidden_size)).to('cuda:0')
         )

    def forward(self, x):
        x = self.emb(x)
        x, (h_n, c_n) = self.lstm(x, self.init_hidden_cell())
        # 1. use sequence result
        if not self.use_cell:
            x = x.view(-1, self.seq_len * self.hidden_size * self.bi_num)
            x = self.fc_seq(x)
        # 2. use hidden cell rsult
        else:
            x = torch.cat((h_n[-1], h_n[-2]), dim=1)
            x = self.fc_cell(x)
        x = self.act1(x)
        x = self.fc(x)
        x = self.act2(x)
        return x
