import torch

from torch import nn


# =========================================================
# 1. 假设词表大小为 10
# 每个 token embedding 维度为 4
# =========================================================

embedding = nn.Embedding(
    num_embeddings=10,
    embedding_dim=4
)


# =========================================================
# 2. 假设一句话被 tokenizer 转成 token IDs
# =========================================================

token_ids = torch.tensor([
    1,
    3,
    5,
    2
])


# =========================================================
# 3. Token ID → Embedding Vector
# =========================================================

vectors = embedding(
    token_ids
)


print("Token IDs:")
print(token_ids)

print("\nEmbedding:")
print(vectors)

print("\nShape:")
print(vectors.shape)