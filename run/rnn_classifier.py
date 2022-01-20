import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from data.names import NamesDataset
from nets.RnnNets import RnnClassifier
from libs.utils import time_since, show_plot


def create_tensor(t):
    if opt.device != 'cpu':
        t = t.to(opt.device)
    return t


def name2list(name):
    return list(map(ord, name)), len(name)


def make_tensor(names, classes):
    seq_names_length = [name2list(name) for name in names]
    seq_names = [x[0] for x in seq_names_length]
    seq_lengths = torch.tensor([x[1] for x in seq_names_length])

    seq_tensor = torch.zeros(len(seq_names), seq_lengths.max().item(), dtype=torch.long)
    for idx, (seq, seq_len) in enumerate(seq_names_length):
        seq_tensor[idx, :seq_len] = torch.tensor(seq, dtype=torch.long)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    classes = classes[perm_idx]
    classes = torch.tensor(classes, dtype=torch.long)

    return create_tensor(seq_tensor), seq_lengths, create_tensor(classes)


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(train_loader, 1):
        inputs, seq_lengths, labels = make_tensor(names, countries)
        outputs = classifier(inputs, seq_lengths)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print('[%s] Epoch %d [%d/%d], loss=%.4f' % (
                time_since(begin), epoch, i * len(inputs), train_set.len, total_loss / (i * len(inputs))
            ))
    return total_loss


def testModel():
    correct = 0
    print("Evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loader, 1):
            inputs, seq_lengths, labels = make_tensor(names, countries)
            outputs = classifier(inputs, seq_lengths)
            pred = outputs.max(dim=1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

        print("Test set: Accuracy %d/%d %.2f%%" % (correct, test_set.len, (100 * correct / test_set.len)))
    return correct / test_set.len

def prediction():
    while True:
        name = input('Input a name:')
        if name == 'q' or name == 'quit':
            break
        inputs, seq_len, _ = make_tensor([name], [0])
        outputs = classifier(inputs, seq_len)
        pred = torch.argmax(outputs)
        print('%s suppose to be a %s name' % (name, train_set.idx2Country(pred.detach().item())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=256)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--layer', type=int, default=2, help='number of layers')
    parser.add_argument('--device', type=str, default='cuda', help='use cuda')
    parser.add_argument('--hidden', type=int, default=100, help='number of hidden cells')
    opt = parser.parse_args()

    train_set = NamesDataset()
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False)
    test_set = NamesDataset(False)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

    for i in range(3):
        print(train_set[i])

    exit()
    N_CLASSES = train_set.getCountryNum()
    # chars as ASCII
    INPUT_SIZE = 128

    classifier = RnnClassifier(opt.batch_size, opt.hidden, N_CLASSES, opt.layer)
    if opt.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(opt.device)
        classifier.to(device)

    dict_dir = '../logs/rnn-' + type(classifier).__name__ + '.pth'

    if os.path.exists(dict_dir):
        classifier.load_state_dict(torch.load(dict_dir))
        prediction()
        exit()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    begin = time.time()
    print("Trainning for %d epochs.." % opt.epochs)
    loss_list = []
    acc_list = []

    print(train_set.country_num, train_set.len)

    for epoch in range(opt.epochs):
        # Train loop
        loss = trainModel()
        acc = testModel()
        loss_list.append(loss)
        acc_list.append(acc)

    torch.save(classifier.state_dict(), dict_dir)

    show_plot(loss_list, 'Loss', 'Names & Countries')
    show_plot(acc_list, 'Accuracy', 'Names & Countries')
