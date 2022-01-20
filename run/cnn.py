import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from nets.CnnNets import CnnNet1, CnnNet2, GoogleNet, ResidualNet

# in_ch, out_ch = 5, 10
# width, height = 100, 100
#
# kernel_size = 3
#
# input = torch.randn(batch_size, in_ch, width, height)
# conv_layer = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size)
#
# output = conv_layer(input)
#
# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)


batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化（正态分布）
])

train_set = datasets.MNIST(root='../data/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_set = datasets.MNIST(root='../data/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)


# model = CnnNet1()
# model = CnnNet2()
# model = GoogleNet()
model = ResidualNet()
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
dict_dir = '../logs/cnn-' + type(model).__name__ + '.pth'
if os.path.exists(dict_dir):
    model.load_state_dict(torch.load(dict_dir))

# train

def train(epoch_idx):
    epoch_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # input, target
        # sample, label
        optimizer.zero_grad()
        x, y = data
        # 送入gpu
        x = x.to(device)
        y = y.to(device)

        # forward
        pred = model(x)
        # 计算loss
        loss = loss_fn(pred, y)

        # backward
        loss.backward()
        # 更新学习率
        optimizer.step()
        # 整一批次的loss
        epoch_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch_idx + 1, batch_idx + 1, epoch_loss / 100))
            epoch_loss = 0


def test():
    """
    验证模型
    :return:
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


def predict(n):
    """
    泛化预测
    :return:
    """
    x, y = test_loader.dataset[n]
    outputs = model(torch.unsqueeze(x, 0).to(device))
    _, predicted = torch.max(outputs.data, dim=1)
    return predicted.item(), y


if __name__ == '__main__':
    print(f"Using {device} device")

    if not os.path.exists(dict_dir):
        for epoch in range(10):
            train(epoch)
            test()
        print('training finish.')
        torch.save(model.state_dict(), dict_dir)
    else:
        print('There is already a trained mode!')
        # test()
    accuracy = 0
    cnt = 20
    for i in range(cnt):
        n = random.randint(0, 5000)
        pred, label = predict(n)
        print(f'%d. pic No.%d is %d, and the model predicts %d' % (i, n, label, pred))
        if pred == label:
            accuracy += 1
    print(f'All were done, total accuracy is %.1f%%' % (accuracy * 100 / cnt))
