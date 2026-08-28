from sentence_transformers import SentenceTransformer
import numpy as np

# 加载支持中文的 Embedding 模型
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

texts = [
    "我喜欢学习人工智能",
    "人工智能和机器学习是热门技术",
    "我正在学习大模型相关知识",
    "今天天气很好，适合出去散步",
    "如何让大模型查询外部资料？"
]


# 对4条文本进行 Embedding
embeddings = model.encode(texts)

# 自己实现余弦相似度
def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

# 每一条文本轮流作为 query
for query_index in range(len(texts)):

    query = texts[query_index]
    query_embedding = embeddings[query_index]

    print(f"\n===== Query：{query} =====")

    # 和剩余4条文本进行比较
    for text_index in range(len(texts)):

        # 不和自己比较
        if text_index == query_index:
            continue

        similarity = cosine_similarity(
            query_embedding,
            embeddings[text_index]
        )

        print(f"对照文本：{texts[text_index]}")
        print(f"相似度：{similarity:.4f}")
        print()