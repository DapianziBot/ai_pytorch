import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from libs.IMDB import get_dataloader, ws, MAX_SEQ_LEN
from libs.utils import ROOT
from nets.RnnNets import IMDBReview

device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

IS_TRAIN = True
WORD_COUNT = len(ws)
BATCH_SIZE = 256
CLASSES = 2
SEQ_LEN = MAX_SEQ_LEN
DROPOUT = 0.5

model_state = ROOT + '/logs/rnn-lstm-bi-imdb.model.pkl'
optim_state = ROOT + '/logs/rnn-lstm-bi-imdb.optim.pkl'

train_loader = get_dataloader(batch_size=BATCH_SIZE)
test_loader = get_dataloader(False, batch_size=BATCH_SIZE)

model = IMDBReview(WORD_COUNT, SEQ_LEN, BATCH_SIZE, CLASSES, dropout=DROPOUT).to(device)


def train(epoch):
    loss_list = []
    acc_list = []
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit_scale=True)
    for i, (x, y) in pbar:
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        optim.zero_grad()
        y_ = model(x)

        loss = loss_fn(y_, y)

        loss.backward()
        optim.step()

        acc = torch.argmax(y_, dim=-1).eq(y).float().mean()
        pbar.set_description("Epoch %d: loss=%.4f, acc=%.1f%%" % (epoch + 1, loss.item(), acc.item()*100))
        loss_list.append(loss.detach().item())
        acc_list.append((acc.detach().item()))
    return loss_list, acc_list


def eval(epoch):
    loss_list = []
    acc_list = []
    model.eval()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), unit_scale=True)
    for idx, (x, y) in pbar:
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        with torch.no_grad():
            output = model(x)
            cur_loss = loss_fn(output, y)
            loss_list.append(cur_loss.detach().item())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(y).float().mean().detach().item()
            acc_list.append(cur_acc)
        pbar.set_description("Epoch %d: Test loss=%.4f, Test acc=%.1f%%" % (epoch + 1, loss_list[-1], acc_list[-1]*100))
    return loss_list, acc_list


if os.path.exists(model_state) and not IS_TRAIN:
    model.load_state_dict(model_state)
else:
    loss_fn = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(10):
        t_l, t_a = train(epoch)
        e_l, e_a = eval(epoch)
        # print('Epoch %d done: loss=%.4f' % (epoch, loss))
        train_loss.extend(t_l)
        train_acc.extend(t_a)
        test_loss.extend(e_l)
        test_acc.extend(e_a)

    torch.save(model.state_dict(), model_state)
    torch.save(optim.state_dict(), optim_state)

    plt.figure(figsize=(8, 12))
    plt.subplot(211)
    plt.plot(train_acc, color='red', label='Train Accuracy')
    plt.plot(test_acc, color='blue', label='Test dataset Acc')
    plt.title('IMDB review with LSTM Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(212)
    plt.plot(train_loss, color='red', label='Train Loss')
    plt.plot(test_loss, color='blue', label='Test dataset Loss')
    plt.title('IMDB review with LSTM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
