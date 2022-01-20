import os
import pickle

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from libs.utils import tokenize, ROOT
from libs.word2sequence import Word2Sequence

MAX_SEQ_LEN = 100

class IMDBSentiment(Dataset):
    label_dict = {'neg': 1, 'pos': 0}

    def __init__(self, train=True):
        super(IMDBSentiment, self).__init__()

        self.data = []
        self.labels = []
        self.file_dir = ROOT + "\\data\\aclImdb\\" + ('train' if train else 'test')

        # neg
        self.read_dir('neg')
        # pos
        self.read_dir('pos')
        self.len = len(self.labels)

    def read_dir(self, flag):
        files = os.listdir(self.file_dir + ('\\%s\\' % flag))
        for file in files:
            self.data.append('\\%s\\%s' % (flag, file))
            self.labels.append(self.label_dict[flag])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        file = self.file_dir + self.data[item]
        with open(file, 'r', encoding='utf-8') as f:
            sentiment = tokenize(f.read().strip())

        return sentiment, self.labels[item]


def collate_fn(batch):
    """
    :param batch: (__getitem__() * batch)
    :return:
    """
    sentence, label = list(zip(*batch))
    sentence = [ws.transform(x, max_len=MAX_SEQ_LEN) for x in sentence]
    return sentence, label


ws_path = os.path.join(ROOT, r'data\aclImdb\imdb_vocab_5.pkl')

ws = pickle.load(open(ws_path, 'rb'))


def get_dataloader(train=True, batch_size=2):
    data_set = IMDBSentiment(train)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)


if __name__ == '__main__':

    ws = Word2Sequence()
    dataset = IMDBSentiment()
    for i in tqdm(range(len(dataset))):
        comment, _ = dataset[i]
        ws.fit(comment)

    ws.build_vocab(min=5)

    pickle.dump(ws, open(ws_path, 'w+b'))
    print(len(ws))
