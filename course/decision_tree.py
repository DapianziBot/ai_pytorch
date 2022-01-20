import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
X = np.array([
    ['<=30', 'high', 'no', 'fair', 'no'],
    ['<=30', 'high', 'no', 'excellent', 'no'],
    ['31..40', 'high', 'no', 'fair', 'yes'],
    ['>40', 'medium', 'no', 'fair', 'yes'],
    ['>40', 'low', 'yes', 'fair', 'yes'],
    ['>40', 'low', 'yes', 'excellent', 'no'],
    ['31..40', 'low', 'yes', 'excellent', 'yes'],
    ['<=30', 'medium', 'no', 'fair', 'no'],
    ['<=30', 'low', 'yes', 'fair', 'yes'],
    ['>40', 'medium', 'yes', 'fair', 'yes'],
    ['<=30', 'medium', 'yes', 'excellent', 'yes'],
    ['31..40', 'medium', 'no', 'excellent', 'yes'],
    ['31..40', 'high', 'yes', 'fair', 'yes'],
    ['>40', 'medium', 'no', 'excellent', 'no'],
])

feature_names = ['age', 'income', 'is_student', 'credit_rating']
#

df = pd.DataFrame(X[:, :-1], columns=feature_names)
print(df)

d = dict([(x, i) for i, x in enumerate(set(X[:, -1]))])
y = [d[x] for x in X[:, -1]]
x = pd.get_dummies(df, drop_first=True)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x, y)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=list(x.columns),
                                class_names=['buy_computer_yes', 'buy_computer_no'],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("iris.graph", view=True)
