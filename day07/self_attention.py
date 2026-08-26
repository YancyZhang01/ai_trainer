import math

import torch

from torch import nn


# =========================================================
# 1. 假设有3个token
# 每个token embedding为4维
# =========================================================

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0]
])


embedding_dim = 4


# =========================================================
# 2. 创建 Q / K / V 线性映射
# =========================================================

W_q = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False
)

W_k = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False
)

W_v = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False
)


# =========================================================
# 3. 得到 Q / K / V
# =========================================================

Q = W_q(X)
K = W_k(X)
V = W_v(X)


# =========================================================
# 4. Attention Score
#
# Q @ K.T
# =========================================================

scores = (
    Q @ K.T
)


# =========================================================
# 5. Scaled Dot-Product
# =========================================================

scores = (
    scores
    / math.sqrt(embedding_dim)
)


# =========================================================
# 6. Softmax 转为权重
# =========================================================

attention_weights = torch.softmax(
    scores,
    dim=-1
)


# =========================================================
# 7. 对 V 做加权求和
# =========================================================

output = (
    attention_weights
    @ V
)


print("Input:")
print(X)

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

print("\nAttention Scores:")
print(scores)

print("\nAttention Weights:")
print(attention_weights)

print("\nOutput:")
print(output)