import faiss
from sentence_transformers import SentenceTransformer


# =========================
# 1. 准备中文 document
# =========================
documents = [
    "人工智能正在改变我们的工作方式。",
    "机器学习是人工智能的重要分支。",
    "深度学习通常使用多层神经网络。",
    "自然语言处理可以让计算机理解人类语言。",
    "Python 是人工智能开发中常用的编程语言。",
    "FAISS 是一个高效的向量相似性搜索库。",
    "Sentence Transformers 可以将文本转换为语义向量。",
    "向量数据库可以用于存储和检索文本向量。",
    "语义搜索关注文本含义，而不仅仅是关键词匹配。",
    "大语言模型能够生成和理解自然语言。",
    "今天天气很好，适合出去散步。",
    "北京是中国的首都。",
    "上海是中国重要的金融中心。",
    "新能源汽车正在快速发展。",
    "苹果是一种常见的水果。",
    "篮球是一项非常受欢迎的体育运动。",
    "数据库可以用于保存和管理大量数据。",
    "推荐系统可以根据用户兴趣推荐内容。",
    "计算机视觉主要研究如何让机器理解图片和视频。",
    "搜索引擎帮助用户从大量信息中找到需要的内容。",
    "Embedding 可以将文本表示成高维数值向量。",
    "余弦相似度经常用于衡量两个文本向量的语义相似程度。",
]


# =========================
# 2. 加载 SentenceTransformer
# 强制使用 CPU
# =========================
model_name = "BAAI/bge-small-zh-v1.5"

model = SentenceTransformer(
    model_name,
    device="cpu"
)

print("模型加载完成")
print("document 数量:", len(documents))


# =========================
# 3. document embedding
# =========================
document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\n=== Document Embedding ===")
print("document_embeddings.shape:", document_embeddings.shape)


# =========================
# 4. 创建 FAISS index
# =========================
embedding_dimension = document_embeddings.shape[1]

print("embedding_dimension:", embedding_dimension)

# 因为 embedding 已经归一化，
# Inner Product 就等价于 Cosine Similarity
index = faiss.IndexFlatIP(embedding_dimension)

index.add(document_embeddings)

print("\n=== FAISS Index ===")
print("FAISS index 中向量数量:", index.ntotal)


# =========================
# 5. 用户输入 query
# =========================
query = input("\n请输入你的搜索问题：")


# =========================
# 6. query embedding
# =========================
query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\n=== Query Embedding ===")
print("query_embedding.shape:", query_embedding.shape)


# =========================
# 7. FAISS 相似性检索
# =========================
top_k = 3

scores, indices = index.search(
    query_embedding,
    top_k
)

print("\n=== Search Result Shape ===")
print("scores.shape:", scores.shape)
print("indices.shape:", indices.shape)
print(
    "Index Total:",
    index.ntotal
)


# =========================
# 8. 输出 Top 3
# =========================
print("\n==========================")
print("Query:", query)
print("Top 3 Semantic Search Results")
print("==========================")

for rank, (doc_index, score) in enumerate(
    zip(indices[0], scores[0]),
    start=1
):
    print(f"\nRank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Document: {documents[doc_index]}")