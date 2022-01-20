import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np

cwd = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(cwd, '..'))


def time_since(t):
    s = time.time() - t
    m = s // 60
    s = s % 60
    return '%d分%d秒' % (m, s)


def show_plot(data, label, title):
    plt.title(title)
    plt.plot(np.array(range(1, len(data) + 1)), data)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.tight_layout()
    plt.show()

    # plt.savefig(
    #     '%s/charts/%s-%s-%s.png' % (cwd, title, label, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(int(time.time()))))
    # )


def tokenize(text):
    # 过滤其他字符
    text = re.sub(r'[^\w\s\d\.,\':;\?\!]|_', ' ', text)
    # 处理缩写
    text = re.sub(r'\b\'s', ' is', text)
    text = re.sub(r'\b\'m', ' am', text)
    text = re.sub(r'\b\'re', ' are', text)
    text = re.sub(r'n\'t', ' not', text)
    text = re.sub(r'(?=[\.,\':;\?\!])', ' ', text)
    # 小写
    return [word.strip().lower() for word in text.split()]
