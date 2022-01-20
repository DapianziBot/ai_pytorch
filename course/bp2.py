import torch

x = torch.tensor([1.0, 0, 1.0])

lr = 0.9
neural = [2, 1]

w1 = torch.tensor([
    [0.2, -0.3],
    [0.4, 0.1],
    [-0.5, 0.2]
]).t()
b1 = torch.tensor([-0.4, 0.2])
w2 = torch.tensor([[-0.3], [-0.2]]).t()
b2 = torch.tensor([0.1])

truth = torch.tensor([1.])

class BPDemo(torch.nn.Module):
    def __init__(self):
        super(BPDemo, self).__init__()
        self.fc1 = torch.nn.Linear(3, 2)
        self.fc2 = torch.nn.Linear(2, 1)

        self.weights_initialization()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def weights_initialization(self):
        '''
        When we define all the modules such as the layers in '__init__()'
        method above, these are all stored in 'self.modules()'.
        We go through each module one by one. This is the entire network,
        basically.
        '''

        self.fc1.weight = torch.nn.Parameter(w1.clone())
        self.fc1.bias = torch.nn.Parameter(b1.clone())
        self.fc2.weight = torch.nn.Parameter(w2.clone())
        self.fc2.bias = torch.nn.Parameter(b2.clone())

model = BPDemo()

loss_fn = torch.nn.L1Loss()
optm = torch.optim.SGD(model.parameters(), lr=0.9)

for i in range(1, 3):
    print(model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy())
    print(model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy())
    out = model(x)

    optm.zero_grad()
    print(out)
    loss = loss_fn(truth, out)
    print(loss.detach().item())
    loss.backward()
    optm.step()

print('done')
