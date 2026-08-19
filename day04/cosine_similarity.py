import numpy as np


def cosine_similarity(a, b):

    # np.dot(a, b)
    #
    # 计算两个向量点积

    dot_product = np.dot(a, b)


    # np.linalg.norm(a)
    #
    # 计算向量 a 的长度

    norm_a = np.linalg.norm(a)


    # 计算向量 b 的长度

    norm_b = np.linalg.norm(b)


    # 余弦相似度公式：
    #
    # dot(a,b)
    # ----------------
    # norm(a) * norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)


    return similarity
