import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import ROOT_PATH

print(f'Pytorch version: {torch.__version__}')

data_path = ROOT_PATH + "/data"
training_data = datasets.FashionMNIST(
    root=data_path,
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root=data_path,
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_data_loader = DataLoader(training_data, batch_size=batch_size)
test_data_loader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_data_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class DemoNeuralNetwork(nn.Module):
    def __init__(self):
        super(DemoNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = DemoNeuralNetwork().to(device)
print(model)

# 交叉熵
loss_fn = nn.CrossEntropyLoss()
# 随机梯度下降法
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 预测偏差
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy:  {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n{'=' * 50}")
    train(train_data_loader, model, loss_fn, optimizer)
    test(test_data_loader, model)
print("Done!")

path = ROOT_PATH + "/models/fashion_MNIST_demo.pth"
torch.save(model.state_dict(), path)
print(f"Saved PyTorch Model State to '{path}'")

model = DemoNeuralNetwork()
model.load_state_dict(torch.load(path))

classes = [
    "T-shirt/top",
    "裤子",
    "连帽衫",
    "裙子",
    "外套",
    "拖鞋",
    "衬衫",
    "运动鞋",
    "背包",
    "短靴",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"识别为: '{predicted}', 实际是: '{actual}'")
