import torch

from nets.RnnNets import SimpleRnn, SimpleRnn2

num_class=4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]
one_hot = [one_hot_lookup[x] for x in x_data]
# 5,1,4
inputs = torch.tensor(x_data).view(batch_size, seq_len)
# 5, 1
labels = torch.tensor(y_data)
# labels = torch.tensor([one_hot_lookup[x] for x in y_data])

# net = SimpleRnn(input_size, hidden_size, batch_size)
net = SimpleRnn2(input_size, hidden_size, batch_size, num_class, embedding_size, num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

epochs = 20
#
# for epoch in range(epochs):
#     loss = 0
#     optimizer.zero_grad()
#     hidden = net.init_hidden()
#     print('Predicted string:', end='')
#     for input, label in zip(inputs, labels):
#         hidden = net(input, hidden)
#         # TODO
#         loss += criterion(hidden, label)
#         _, idx = hidden.max(dim=1)
#         print(idx2char[idx.item()], end='')
#
#     loss.backward()
#     optimizer.step()
#     print(', Epoch [%d/%d] loss=%.4f' % (epoch + 1, epochs, loss.item()))

for epoch in range(epochs):
    loss = 0
    # 初始化优化器
    optimizer.zero_grad()
    # 输出，隐层
    output = net(inputs)
    # 计算交叉熵
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    idx = torch.argmax(output, dim=1)
    idx = idx.numpy()

    print('Predicted string:', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/%d] loss=%.4f' % (epoch + 1, epochs, loss.item()))
