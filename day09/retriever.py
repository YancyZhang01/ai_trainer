import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 准备结构化中文 documents
# ============================================================

documents = [
    {
        "document_id": "doc_001",
        "category": "人工智能",
        "source": "AI基础知识",
        "text": "人工智能正在改变我们的工作方式。"
    },
    {
        "document_id": "doc_002",
        "category": "人工智能",
        "source": "AI基础知识",
        "text": "机器学习是人工智能的重要分支。"
    },
    {
        "document_id": "doc_003",
        "category": "深度学习",
        "source": "AI基础知识",
        "text": "深度学习通常使用多层神经网络学习复杂的数据特征。"
    },
    {
        "document_id": "doc_004",
        "category": "自然语言处理",
        "source": "NLP教程",
        "text": "自然语言处理可以让计算机理解和处理人类语言。"
    },
    {
        "document_id": "doc_005",
        "category": "编程",
        "source": "Python教程",
        "text": "Python 是人工智能开发中常用的编程语言。"
    },
    {
        "document_id": "doc_006",
        "category": "向量检索",
        "source": "FAISS教程",
        "text": "FAISS 是一个高效的向量相似性搜索库。"
    },
    {
        "document_id": "doc_007",
        "category": "Embedding",
        "source": "Embedding教程",
        "text": "Sentence Transformers 可以将文本转换为具有语义信息的向量。"
    },
    {
        "document_id": "doc_008",
        "category": "向量数据库",
        "source": "向量数据库教程",
        "text": "向量数据库可以用于存储和检索文本、图片等数据产生的向量。"
    },
    {
        "document_id": "doc_009",
        "category": "语义搜索",
        "source": "搜索技术教程",
        "text": "语义搜索关注文本表达的含义，而不仅仅进行关键词匹配。"
    },
    {
        "document_id": "doc_010",
        "category": "大语言模型",
        "source": "LLM教程",
        "text": "大语言模型能够理解和生成自然语言。"
    },
    {
        "document_id": "doc_011",
        "category": "生活",
        "source": "生活资讯",
        "text": "今天天气很好，非常适合出去散步。"
    },
    {
        "document_id": "doc_012",
        "category": "地理",
        "source": "地理知识",
        "text": "北京是中华人民共和国的首都。"
    },
    {
        "document_id": "doc_013",
        "category": "地理",
        "source": "地理知识",
        "text": "上海是中国重要的金融和经济中心。"
    },
    {
        "document_id": "doc_014",
        "category": "汽车",
        "source": "汽车资讯",
        "text": "新能源汽车行业近年来正在快速发展。"
    },
    {
        "document_id": "doc_015",
        "category": "水果",
        "source": "生活知识",
        "text": "苹果是一种常见的水果，含有丰富的营养。"
    },
    {
        "document_id": "doc_016",
        "category": "体育",
        "source": "体育资讯",
        "text": "篮球是一项非常受欢迎的团队体育运动。"
    },
    {
        "document_id": "doc_017",
        "category": "数据库",
        "source": "数据库教程",
        "text": "数据库可以用于保存、组织和管理大量结构化数据。"
    },
    {
        "document_id": "doc_018",
        "category": "推荐系统",
        "source": "推荐系统教程",
        "text": "推荐系统可以根据用户的兴趣和行为推荐相关内容。"
    },
    {
        "document_id": "doc_019",
        "category": "计算机视觉",
        "source": "CV教程",
        "text": "计算机视觉主要研究如何让计算机理解图片和视频。"
    },
    {
        "document_id": "doc_020",
        "category": "搜索引擎",
        "source": "搜索技术教程",
        "text": "搜索引擎可以帮助用户从大量信息中快速找到需要的内容。"
    },
    {
        "document_id": "doc_021",
        "category": "Embedding",
        "source": "Embedding教程",
        "text": "Embedding 可以把文本表示成高维数值向量。"
    },
    {
        "document_id": "doc_022",
        "category": "相似度",
        "source": "向量检索教程",
        "text": "余弦相似度常用于衡量两个文本向量之间的语义相似程度。"
    },
]


# ============================================================
# 2. 加载 Embedding 模型
# 强制使用 CPU
# ============================================================

model_name = "BAAI/bge-small-zh-v1.5"

model = SentenceTransformer(
    model_name,
    device="cpu"
)

print("模型加载完成")
print("document 数量:", len(documents))


# ============================================================
# 3. 提取所有 document 的 text
# ============================================================

document_texts = [
    document["text"]
    for document in documents
]

print("\n=== Document Text ===")
print("document_texts 数量:", len(document_texts))


# ============================================================
# 4. 将中文文本库转换成 Embedding
# ============================================================

document_embeddings = model.encode(
    document_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# FAISS 通常使用 float32
document_embeddings = document_embeddings.astype(np.float32)


print("\n=== Document Embeddings ===")
print(
    "document_embeddings.shape:",
    document_embeddings.shape
)


# ============================================================
# 5. 创建 FAISS Index
# ============================================================

embedding_dimension = document_embeddings.shape[1]

print(
    "embedding_dimension:",
    embedding_dimension
)


# normalize_embeddings=True 后
# 使用 Inner Product 可以近似看成 Cosine Similarity
index = faiss.IndexFlatIP(
    embedding_dimension
)


# ============================================================
# 6. 将 document embeddings 加入 FAISS
# ============================================================

index.add(
    document_embeddings
)

print("\n=== FAISS Index ===")
print(
    "index.ntotal:",
    index.ntotal
)


# ============================================================
# 7. 封装 search 函数
# ============================================================
def search(query, top_k=3):
    """
    根据输入 query，在 FAISS 检索库中查找最相关的 top_k 个 document。

    参数：
        query (str): 用户查询文本
        top_k (int): 返回结果数量

    返回：
        list[dict]: Top-K 检索结果
    """

    # 1. 将 query 转换为 embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    print("query_embedding.shape:", query_embedding.shape)

    # 2. 防止 top_k 超过索引中的 document 数量
    top_k = min(top_k, index.ntotal)

    # 3. FAISS 相似性检索
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    print("scores.shape:", scores.shape)
    print("indices.shape:", indices.shape)

    # 4. 根据 FAISS 返回的索引找到原始 document
    results = []

    for rank, (doc_index, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1
    ):
        document = documents[doc_index]

        results.append({
            "rank": rank,
            "score": float(score),
            "document_id": document["document_id"],
            "category": document["category"],
            "source": document["source"],
            "text": document["text"]
        })

    return results