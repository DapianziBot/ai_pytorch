from functools import reduce

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB


def softmax(x):
    f_x = x / np.sum(x)
    return f_x


def naive_bayes(samples, X):
    sample = []
    df = pd.DataFrame(samples)
    for i in df:
        if i == 4:
            break
        df[i] = pd.Categorical(df[i])
        df[i] = df[i].cat.codes

    samples = df.to_numpy()
    [sample.extend([x[:-1]] * x[-1]) for x in samples]
    sample = np.array(sample)
    x = np.array(sample[:, :-1])
    y = np.array(sample[:, -1])
    cnb = CategoricalNB()
    cnb.fit(x, y)

    print(cnb.get_params())

    X = [[3, 1, 4]]
    print(cnb.predict(X))

    print(cnb.predict_proba(X))


def bayes_p(samples, X):
    total = sum(np.array(samples[:, -1], dtype=np.int32))
    labels = set(samples[:, -2])
    c_y = {}
    p_y = {}
    for label in labels:
        c_y[label] = sum([np.array(x[-1], dtype=np.int32) for x in samples if x[-2] == label])
        p_y[label] = c_y[label] / total
        print('P(%s) = %d/%d = %.4f' % (label, c_y[label], total, p_y[label]))
    p_x = []
    for i, x in enumerate(X):
        p_x.append({})
        for label in labels:
            c = sum([np.array(item[-1], dtype=np.int32) for item in samples if (item[i] == x and item[-2] == label)])
            if c == 0:
                p = 1 / (c_y[label] + len(set(samples[:, i])))
                print('P(%s|%s) = 1/(%d + %d) = %.4f -- 拉普拉斯校正' % (x, label, c_y[label], len(set(samples[:, i])), p))
            else:
                p = c / c_y[label]
                print('P(%s|%s) = %d/%d = %.4f' % (x, label, c, c_y[label], p))
            p_x[i][label] = p

    pred_p = []
    for i, label in enumerate(labels):
        pred_p.append(reduce(lambda x, y: x * y, [x[label] for x in p_x], p_y[label]))
        print('P(X|%s) = %.4f' % (label, pred_p[i]))
    pred = tuple(labels)[np.array(pred_p).argmax()]
    # print(softmax(np.array(pred_p)))

    print('Predict: X is', pred)


X = [
    ['强', '无', '优', '无', '不推荐', 1],
    ['强', '无', '优', '有', '不推荐', 1],
    ['一般', '无', '优', '无', '推荐', 1],
    ['弱', '无', '良', '无', '推荐', 1],
    ['弱', '有', '中', '无', '推荐', 1],
    ['弱', '有', '中', '有', '不推荐', 1],
    ['一般', '有', '中', '有', '推荐', 1],
    ['强', '无', '优', '无', '不推荐', 1],
    ['强', '有', '中', '无', '推荐', 1],
    ['弱', '有', '良', '无', '推荐', 1],
    ['强', '有', '良', '有', '推荐', 1],
    ['一般', '无', '优', '有', '推荐', 1],
    ['一般', '有', '优', '无', '推荐', 1],
    ['弱', '无', '良', '有', '不推荐', 1]
]
# X = [
#     ['sales', '31...35', '46k..50k', 'senior', 30],
#     ['sales', '26...30', '26k..30k', 'junior', 40],
#     ['sales', '31...35', '31k..35k', 'junior', 40],
#     ['systems', '21...25', '46k..50k', 'junior', 20],
#     ['systems', '31...35', '66k..70k', 'senior', 5],
#     ['systems', '26...30', '46k..50k', 'junior', 3],
#     ['systems', '41...45', '66k..70k', 'senior', 3],
#     ['marketing', '36...40', '46k..50k', 'senior', 10],
#     ['marketing', '31...35', '41k..45k', 'junior', 4],
#     ['secretary', '46...50', '36k..40k', 'senior', 4],
#     ['secretary', '26...30', '26k..30k', 'junior', 6],
# ]

if __name__ == '__main__':
    # naive_bayes(X, ['systems', '26...30', '46k..50k'])
    bayes_p(np.array(X), ['强','无','良','有'])
