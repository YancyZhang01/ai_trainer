import numpy as np


def skills_to_vector(skills, all_skills):

    # np.zeros(...)
    #
    # 创建一个全0向量。
    #
    # len(all_skills)
    # 表示向量长度等于技能总数。
    #
    # dtype=int
    # 指定元素类型为整数。

    vector = np.zeros(
        len(all_skills),
        dtype=int
    )


    # enumerate(...)
    #
    # 今天第一次出现。
    #
    # 它可以同时给我们：
    # index和value
    #
    # 例如：
    # all_skills =
    # ["python","rag"]
    # enumerate后：
    # 0, "python"
    # 1, "rag"

    for index, skill in enumerate(all_skills):

        # 判断这个技能
        # 是否存在于当前人的技能列表里

        if skill in skills:

            # 如果会这个技能，
            # 对应位置设成 1

            vector[index] = 1


    return vector

