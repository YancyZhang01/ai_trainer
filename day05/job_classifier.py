from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. 读取数据
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILE_PATH = DATA_DIR / "jobs.csv"

df = pd.read_csv(FILE_PATH)

print("===== Dataset =====")
print(df.head())

print("\nLabel Distribution:")
print(df["label"].value_counts())


# =========================================================
# 2. 准备 X 和 y
# =========================================================

X = df["text"]
y = df["label"]


# =========================================================
# 3. 划分训练集和测试集
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTrain Size:", len(X_train))
print("Test Size :", len(X_test))


# =========================================================
# 4. TF-IDF 文本向量化
# =========================================================

vectorizer = TfidfVectorizer()


# 训练集：
# 学习词表 + 转换为数值向量
X_train_vector = vectorizer.fit_transform(
    X_train
)


# 测试集：
# 只能使用训练集已经学好的词表进行转换
X_test_vector = vectorizer.transform(
    X_test
)


print(
    "TF-IDF Train Shape:",
    X_train_vector.shape
)


# =========================================================
# 5. 创建并训练模型
# =========================================================

model = LogisticRegression(
    max_iter=500
)

model.fit(
    X_train_vector,
    y_train
)


# =========================================================
# 6. 测试集预测
# =========================================================

predictions = model.predict(
    X_test_vector
)


# =========================================================
# 7. 模型评估
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions
)


print("\n===== Evaluation =====")

print(
    f"Accuracy : {accuracy:.3f}"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall   : {recall:.3f}"
)

print(
    f"F1       : {f1:.3f}"
)

print("\nConfusion Matrix:")
print(matrix)


# =========================================================
# 8. 测试新的 JD
# =========================================================

new_jobs = [
    "负责 RAG 向量检索 知识库 大模型应用",
    "负责 Java Spring 微服务 MySQL 开发",
    "负责 Agent MCP 工具调用 Prompt 优化"
]


# 新数据只能 transform，不能重新 fit
new_vectors = vectorizer.transform(
    new_jobs
)


new_predictions = model.predict(
    new_vectors
)


print("\n===== New JD Prediction =====")


for text, prediction in zip(
    new_jobs,
    new_predictions
):

    label_name = (
        "LLM应用岗"
        if prediction == 1
        else "非LLM应用岗"
    )

    print("\nJD:", text)

    print(
        "Prediction:",
        label_name
    )


    results = pd.DataFrame({
    "text": X_test,
    "true_label": y_test,
    "prediction": predictions
})


wrong_cases = results[
    results["true_label"]
    != results["prediction"]
]


print("\n===== Wrong Cases =====")
print(wrong_cases)