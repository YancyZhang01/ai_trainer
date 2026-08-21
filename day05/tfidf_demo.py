from sklearn.feature_extraction.text import TfidfVectorizer


texts = [
    "RAG 知识库 向量检索",
    "Agent 大模型 工具调用",
    "Java Spring 后端开发"
]


# TfidfVectorizer()
#
# 创建一个 TF-IDF 文本向量化器。
#
# 注意：
# 此时它还不知道数据里有哪些词。

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(
    texts
)

print(
    vectorizer.get_feature_names_out()
)