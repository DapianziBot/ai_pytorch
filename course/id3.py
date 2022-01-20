from math import log2

import numpy as np


class treeNode():
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []

    def addChild(self, node):
        self.children.append(node)


def printTree(root):
    q = [root]
    while len(q) > 0:
        l = len(q)
        for i in range(l):
            curr = q.pop(0)
            if curr == '|':
                print(curr, end=' ')
                continue
            print(curr.attribute, end=' ')
            for c in curr.children:
                q.append(c)
            q.append('|')
        print('')


def info(x):
    s = sum(x)
    return sum([
        0 if i == 0 else -log2(i / s) * i / s for i in x
    ])


def find_attr_cls(S, idx, r, C, c):
    return len(S[np.where(np.logical_and(S[:, idx] == r, S[:, C] == c))])


def maxGain(R, C, S):
    total = len(S)
    s_c = S[:, C]
    set_c, count_c = np.unique(s_c, return_counts=True)
    info_c = info(count_c)
    print(info_c)
    gains = []
    for idx in R:
        E_r = []
        expr = []
        for r in set(S[:, idx]):
            E_r.append([find_attr_cls(S, idx, r, C, c) for c in set(s_c)])
            print('%s %s: ' % (cols[idx], r), E_r[-1])
            expr.append('%d/%d * I%s' % (sum(E_r[-1]), total, str(set(E_r[-1]))))
        E_r = sum([info(x) * sum(x) / total for x in E_r])
        print('E(%s) = %s = %.3f' % (cols[idx], ' + '.join(expr), E_r))
        gain = info_c - E_r
        gains.append(gain)
    return R[np.array(gains).argmax()]


def id3(R, C, S: list):
    if len(S) == 0:
        return None
    if len(R) == 0:
        return treeNode(cols[C] + ' ' + S[0][C])
    if all([(S[0] == x).all() for x in S]):
        return treeNode(cols[R[0]])

    d = maxGain(R, C, S)
    print('max gain:', cols[d])
    root = treeNode(cols[d])
    R = [x for x in R if x != d]
    for i in set(S[:, d]):
        Si = np.array([x for x in S if x[d] == i])
        root.addChild(id3(R, C, Si))

    return root


cols = ['沟通', '实践', '成绩', '职务', '推荐']

X = [
    ['强', '无', '优', '无', '不推荐'],
    ['强', '无', '优', '有', '不推荐'],
    ['一般', '无', '优', '无', '推荐'],
    ['弱', '无', '良', '无', '推荐'],
    ['弱', '有', '中', '无', '推荐'],
    ['弱', '有', '中', '有', '不推荐'],
    ['一般', '有', '中', '有', '推荐'],
    ['强', '无', '优', '无', '不推荐'],
    ['强', '有', '中', '无', '推荐'],
    ['弱', '有', '良', '无', '推荐'],
    ['强', '有', '良', '有', '推荐'],
    ['一般', '无', '优', '有', '推荐'],
    ['一般', '有', '优', '无', '推荐'],
    ['弱', '无', '良', '有', '不推荐']
]

root = id3([0, 1, 2, 3], 4, np.array(X))

printTree(root)
