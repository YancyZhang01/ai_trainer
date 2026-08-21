from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

# load_iris()
#
# sklearn自带的一个经典分类数据集。
#
# 数据内容：
# 不同鸢尾花的长度、宽度等特征。
#
# 你今天不是为了学花，
# 而是借它理解机器学习流程。

def load_data():

    # 加载数据
    iris = load_iris()


    # X：模型输入特征
    X = iris.data


    # y：模型要预测的目标
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test