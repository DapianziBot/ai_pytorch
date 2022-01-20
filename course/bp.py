import argparse
from math import exp

import numpy as np


def sigmoid(x):
    if isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    return 1 / (1 + exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def f_expression(n_x, n_w, bias=True):
    expr = []
    for i in n_x:
        expr.append('W_%d%d*X_%d' % (i, n_w, i))
    if bias:
        expr.append('B_%d' % n_w)
    return ' + '.join(expr)


def forward(w, b, in_ch, out_ch, x):
    h = x.dot(w) + b
    y = sigmoid(h)
    for i,_ in enumerate(x):
        for j in range(0, len(out_ch)):
            idx = out_ch[j]
            print('x_%d = %s = sigmoid(%.3f) = %.3f' % (idx, f_expression(in_ch, idx), h[i][j], y[i][j]))
    return y


def backward(w, in_ch, out, err, y, is_out=True):
    if is_out:
        result = d_sigmoid(y) * err
        for i in range(0, len(in_ch)):
            j = in_ch[i]
            print('Err_%d = d_sigmoid(x_%d) * Err(T - x_%d) = %.3f*(1-%.3f) * %.3f = %.4f' % (
                j, j, j, y[i], y[i], err[i], result[i]))
    else:
        result = d_sigmoid(y) * err
        result = w * result.T
        for i in range(0, len(in_ch)):
            j = in_ch[i]
            print('Err_%d = d_sigmoid(x_%d) * Err_%d * W_%d%d = %.3f*(1-%.3f) * %.3f * %.3f = %.4f' % (
                j, j, out, j, out, y[0][i], y[0][i], err, w[i][0], result[i][0]))

    return result


def update_weight(x, w, err, in_ch, out_ch):
    w_d = x * err * lr
    w = w + w_d.T
    for i, m in enumerate(in_ch):
        for j, n in enumerate(out_ch):
            print('W_%d%d = W_%d%d + lr * err_%d * X_%d = %.3f' % (
                m, n, m, n, n, m, w[i][j]))
    return w




lr = 0.9
neural = [2, 1]

weight1 = ([
               [0.1, -0.2],
               [0.3, 0.1],
               [-0.4, 0.1]
           ], [-0.5, 0.1])
weight2 = ([[-0.2], [-0.1]], [0.1])

truth = [1]

inputs = [[0, 1, 0]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=1, help='iterate times for bp')
    opt = parser.parse_args()

    iters = opt.iters

    print('====== Begin ======')

    idx = 1
    for n in inputs[0]:
        print('intput x_%s = %.1f' % (idx, n))
        idx += 1

    x = np.array(inputs)
    w1 = np.array(weight1[0])
    b1 = np.array(weight1[1])

    w2 = np.array(weight2[0])
    b2 = np.array(weight2[1])
    truth = np.array(truth)

    for n in range(1, iters + 1):
        print('########## iteration %d: ' % (n,))
        print('Start forward:')

        h1 = forward(w1, b1, (1, 2, 3), (4, 5), x)

        out = forward(w2, b2, (4, 5), (6,), h1)

        print('Backward: ')
        err_t = truth - out

        err_2 = backward(1, (6,), None, err_t, out)

        err_1 = backward(w2, (4, 5), 6, err_2, h1, False)

        print('\nUpdate Weights: \n')
        update_weight(h1, w2, err_2, (4, 5), (6,))
        update_weight(x, w1, err_1, (1, 2, 3), (4, 5))

        b2 = b2 + lr * err_2
        print('b_6 = %.3f' % tuple(b2))

        b1 = b1 + lr * err_1.reshape(2,)
        print('b_4 = %.3f\nb_5 = %.3f' % tuple(b1))
        idx -= 1
        print('############# iteration %d Finished ################' % (n,))

    print('=======DONE=======')
