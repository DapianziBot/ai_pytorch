"""
将单词转换成向量序列
以及由向量转换成单词
"""


class Word2Sequence:
    """
    """

    # 字典未统计到的词
    UNK_TAG = 'UNK'
    # 句子长度不一，需要对短句进行填充
    PAD_TAG = 'PAD'

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD,
        }
        self.inverse_dict = {}
        self.count = {}


    def __len__(self):
        return len(self.dict)

    def fit(self, sentence):
        """
        把单个句子保存到dict
        :param sentence:
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        """
        生成词典
        :param min: 最小出现次数
        :param max: 最大出现次数
        :param max_features: 一共保留多少单词
        :return:
        """
        # 删除count中词频小于min,大于max的单词
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value >= min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value <= max}

        if max_features is not None:
            self.count = dict(
                # count 中的items按次数倒序，切片
                sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            )

        for word in self.count:
            self.dict[word] = len(self.dict)  # 以当前长度作为字典的值

        # 翻转的字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence: list, max_len=None):
        """
        句子转序列
        :param max_len: int 对句子进行填充
        :param sentence: list
        :return:
        """

        if max_len is not None:
            sen_len = len(sentence)
            if sen_len < max_len:
                sentence.extend([self.PAD_TAG] * (max_len - sen_len))
            if sen_len > max_len:
                sentence = sentence[:max_len]
        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        序列转句子
        :param indices:
        :return:
        """
        return [self.inverse_dict.get(idx, self.UNK_TAG) for idx in indices]


if __name__ == "__main__":
    ws = Word2Sequence()
    sentences = [
        ['我', '爱', '北京', '天安门'],
        ['我', '是', '中国', '人'],
    ]

    for sentece in sentences:
        ws.fit(sentece)

    ws.build_vocab(min=1)
    indices = ws.transform(['我', '是', '北京', '人'])
    print(indices)
    print(ws.inverse_transform(indices))
