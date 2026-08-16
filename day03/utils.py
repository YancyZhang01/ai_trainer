def normalize_skill(skill):
    """
    技能名称标准化

    例如：
    "   Python   "

    变成：

    "python"
    """

    return skill.strip().lower()



def filter_by_city(jobs, city):
    """
    根据城市筛选岗位
    """

    result = []


    for job in jobs:

        if job["city"] == city:

            result.append(job)


    return result