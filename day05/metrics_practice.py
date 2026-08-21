from first_classifier import load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 获取数据
X_train, X_test, y_train, y_test = load_data()


# 创建模型
model = DecisionTreeClassifier()


# 训练
model.fit(
    X_train,
    y_train
)


# 预测
predictions = model.predict(
    X_test
)


# 评价
accuracy = accuracy_score(
    y_test,
    predictions
)


print("准确率:", accuracy)

# 召回率

recall = recall_score(
    y_test,
    predictions,
    average="macro"        #三分类问题需要对recall_score进行平均计算
)


print("召回率:", recall)

f1 = f1_score(
    y_test,
    predictions,
     average="macro"
)
print("f1:", f1)