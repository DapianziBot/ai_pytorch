# -*- coding: utf-8 -*-

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# shopping_list = [['豆奶','莴苣'],
#                  ['莴苣','尿布','葡萄酒','甜菜'],
#                  ['豆奶','尿布','葡萄酒','橙汁'],
#                  ['莴苣','豆奶','尿布','葡萄酒'],
#                  ['莴苣','豆奶','尿布','橙汁']]  # 商店交易信息
shopping_list = [
    ['A', 'C', 'D'],
    ['B', 'C', 'E'],
    ['B', 'E'],
    ['A', 'B', 'C', 'E'],
]  # 商店交易信息

te = TransactionEncoder()  # 使用TransactionEncoder将原始数据转为mlxtend接受的特定数据格式
df_tf = te.fit_transform(shopping_list)  # 转为布尔值的array
# print(df_tf)

df = pd.DataFrame(df_tf, columns=te.columns_)  # 转为dataframe形式，将列名转化为原来的商品名
print(df)

# 设置最小支持度min_support=0.4求频繁项集，use_colnames表示使用商品品表示项目
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
# 设置按照支持度从大到小排序
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
print(frequent_itemsets)

# 设置使用最小置信度=0.9来求关联规则
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
# 设置按照置信度重大到小排序
rules.sort_values(by='confidence', ascending=False, inplace=True)
show_rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
print(show_rules)