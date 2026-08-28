from sentence_transformers import SentenceTransformer

# 强制使用 CPU，避免当前 PyTorch 与 RTX 5070 的 CUDA 兼容问题
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

texts = [
    "我喜欢学习人工智能",
    "人工智能正在改变世界",
    "今天天气很好"
]

# 将文本转换成 Embedding
embeddings = model.encode(texts) 

print("Embedding Python类型：", type(embeddings))
print("Embedding shape：", embeddings.shape)

# 第一条文本的前10维
print("第一条文本的前10维：")
print(embeddings[0][:10])