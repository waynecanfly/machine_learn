'''
使用逻辑回归模型搭建客户流失预警模型
'''

# 1.读取数据
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_excel('D:\迅雷下载\源代码汇总-2020-06-04\第4章 逻辑回归模型\源代码汇总_Jupyter Notebook格式（推荐）\股票客户流失.xlsx')
# 2.划分特征变量与目标变量
X = df.drop(columns='是否流失')
y = df['是否流失']

# 3.划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_size=1) # 数据集的划分默认是随机的，这里random_size相当于指定`种子`的值 使得每次的划分是一致的

# 4.模型搭建
model = LogisticRegression()
model.fit(X_train, y_train)

# 5.模型使用：预测数据结果
y_pred = model.predict(X_test)
print(y_pred[0:100])

# 计算全部数据的预测结果
score = accuracy_score(y_pred, y_test)  # 计算预测值与实际值的准确度
print(score)

# 模型使用：预测概率
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[0:5])



