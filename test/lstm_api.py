import torch
import torch.nn as nn

batch_size = 10
seq_length = 20
word_dict = 100
embedding_dim = 30

inp = torch.randint(low=1, high=100, size=(batch_size, seq_length))
embedding = nn.Embedding(word_dict, embedding_dim=embedding_dim)
input_embedded = embedding(inp)

lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=4, num_layers=1, batch_first=True)
lstm2 = nn.LSTM(input_size=embedding_dim, hidden_size=4, num_layers=2, batch_first=True)
lstm3 = nn.LSTM(input_size=embedding_dim, hidden_size=4, num_layers=1, batch_first=True, bidirectional=True)

# output1, (h1, c1) = lstm1(input_embedded)
# output2, (h2, c2) = lstm2(input_embedded)
output3, (h3, c3) = lstm3(input_embedded)


# print(output1[:, -1, :] == h1[-1, :, :])
# print(output2[:, -1, :] == h2[-1, :, :])
print(output3[:, -1, :4])
print(h3[-2, :, :])
