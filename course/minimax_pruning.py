import math


class TreeNode:
    def __init__(self, name, val=None):
        self.children = []
        self.name = name
        self.miniMaxVal = val
        self.isLeaf = True

    def addChild(self, node):
        self.children.append(node)
        if self.isLeaf:
            self.isLeaf = False


def buildTree(nodeList: list, n=2):
    root = None
    queue = []
    if len(nodeList) > 0:
        tmp = nodeList.pop(0)
        root = TreeNode(*tmp)
        queue.append(root)
    while len(nodeList) > 0:
        l = len(queue)
        for i in range(l):
            p = queue.pop(0)
            for j in range(n):
                if len(nodeList) == 0:
                    return root
                tmp = nodeList.pop(0)
                x = TreeNode(*tmp)
                p.addChild(x)
                queue.append(x)

    return root


def miniMaxPruning(node: TreeNode, depth, alpha=-math.inf, beta=math.inf, maximizing=True):
    if depth == 0 or node.isLeaf:
        return node.miniMaxVal

    if maximizing:
        # Alpha
        maxVal = -math.inf
        for n in node.children:
            if n is None:
                pass

            if alpha >= beta:
                print('#### %s >= %s, [%s]发生β剪枝 ####' % (alpha, beta, n.name))
                continue
            maxVal = max(maxVal, miniMaxPruning(n, depth - 1, alpha, beta, False))
            alpha = max(maxVal, alpha)
            print('[%s] 更新 极小化极大值 ' % (node.name,), maxVal)
        return maxVal

    else:
        # Beta
        minVal = math.inf
        for n in node.children:
            if n is None:
                pass

            if alpha >= beta:
                print('#### %s >= %s, [%s]发生α剪枝 ####' % (alpha, beta, n.name))
                continue
            minVal = min(minVal, miniMaxPruning(n, depth - 1, alpha, beta, True))
            print('[%s] 更新 极小化极大值 ' % (node.name,), minVal)
            beta = min(beta, minVal)
        return minVal


def printTree(t: TreeNode):
    if t is not None:
        print(t.name, t.miniMaxVal)
        for x in t.children:
            printTree(x)


t = buildTree(list(zip(
    list(range(1, 32)),
    ([None] * 15) + [
        # 10, 5, 7, 11, 12, 8, 9, 8, 5, 12, 11, 12, 9, 8, 7, 10
        # 8, 7, 3, 9, 9, 8, 3, 4, 1, 8, 8, 9, 9, 9, 8, 4
        # 0, 5, -3, 3, 3, 6, -2, 3, 5, 4, -3, 0, 6, 8, 9, -3
        8, 5, 6, -4, 3, 8, 4, -6, 1, -9999, 5, 2, 0, 6, 8, 9
    ]
)))

# t = buildTree(list(zip(
#     list(range(1, 16)),
#     ([None] * 7) + [
#         -1, 3, 5, 1, -6, -4, 0, 9
#     ]
# )))

miniMaxPruning(t, 5)
