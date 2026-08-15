import json


# =========================================================
# 功能 1：读取 JSON 文件
# =========================================================

def load_jobs(file_path):
    """
    参数：
        file_path：JSON 文件路径，例如 "jobs.json"

    返回：
        jobs：岗位列表
    """

    # 以读取模式打开 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:

        # json.load()：
        # 把 JSON 文件中的内容转换成 Python 数据
        jobs = json.load(f)

    # 把读取到的岗位列表返回给调用这个函数的地方
    return jobs


# =========================================================
# 功能 2：按照城市筛选岗位
# =========================================================

def filter_by_city(jobs, city):
    """
    参数：
        jobs：所有岗位
        city：想筛选的城市，例如 "杭州"

    返回：
        result：符合城市条件的岗位列表
    """

    # 创建一个空列表，用来保存符合条件的岗位
    result = []

    # 一个岗位一个岗位进行检查
    for job in jobs:

        # job 是一个字典
        # job["city"] 可以取得这个岗位对应的城市
        if job["city"] == city:

            # 如果城市符合要求，就加入 result
            result.append(job)

    # 循环结束后返回筛选结果
    return result


# =========================================================
# 功能 3：统计技能出现次数
# =========================================================

def count_skills(jobs):
    """
    统计所有岗位中，每一种技能出现了多少次。

    返回：
        skill_count：技能频率字典

    例如：
        {
            "Python": 2,
            "RAG": 2,
            "Git": 2
        }
    """

    # 创建空字典
    # key 保存技能名称
    # value 保存出现次数
    skill_count = {}

    # 第一层循环：遍历所有岗位
    for job in jobs:

        # job["skills"] 是一个列表
        # 例如：
        # ["Python", "RAG", "LLM", "Git"]

        # 第二层循环：遍历这个岗位要求的所有技能
        for skill in job["skills"]:

            # 判断这个技能以前有没有统计过
            if skill in skill_count:

                # 如果已经出现过，就在原来的基础上 +1
                skill_count[skill] += 1

            else:

                # 如果第一次出现，就把次数设置为 1
                skill_count[skill] = 1

    return skill_count


# =========================================================
# 功能 4：计算一个岗位的技能匹配率
# =========================================================

def calculate_match_rate(job, my_skills):
    """
    参数：
        job：一个岗位
        my_skills：我目前掌握的技能列表

    返回：
        match_rate：匹配率，例如 0.5

    例如：

    我的技能：
        ["Python", "Git"]

    岗位技能：
        ["Python", "RAG", "LLM", "Git"]

    匹配技能：
        Python
        Git

    所以：
        2 / 4 = 0.5 = 50%
    """

    # 取得这个岗位要求的技能
    job_skills = job["skills"]

    # 如果这个岗位没有填写技能要求，
    # 为了避免后面出现 0 作除数的问题，直接返回 0
    if len(job_skills) == 0:
        return 0

    # matched_count 用来记录：
    # 我的技能中有多少个符合岗位要求
    matched_count = 0

    # 遍历岗位要求的所有技能
    for skill in job_skills:

        # 判断我是否掌握这个技能
        if skill in my_skills:

            # 如果掌握，匹配数量 +1
            matched_count += 1

    # 匹配率 =
    # 匹配上的技能数量 / 岗位要求的技能总数量
    match_rate = matched_count / len(job_skills)

    return match_rate


# =========================================================
# 功能 5：寻找匹配度最高的岗位
# =========================================================

def find_best_job(jobs, my_skills):
    """
    遍历所有岗位，
    找到与自己技能匹配率最高的那个岗位。

    返回：
        best_job：最佳岗位
        best_rate：最佳匹配率
    """

    # 一开始还不知道哪个岗位最好
    best_job = None

    # 当前最高匹配率先设为 0
    best_rate = 0

    # 一个岗位一个岗位计算
    for job in jobs:

        # 调用前面已经写好的函数计算匹配率
        rate = calculate_match_rate(job, my_skills)

        # 如果当前岗位的匹配率
        # 比之前记录的最高匹配率还高
        if rate > best_rate:

            # 更新最高匹配率
            best_rate = rate

            # 同时记住当前岗位
            best_job = job

    # 循环结束以后，
    # best_job 就是匹配度最高的岗位
    return best_job, best_rate


# =========================================================
# 主程序
# =========================================================

# 这里写的是你自己的技能
my_skills = [
    "Python",
    "Git"
]


# ---------------------------------------------------------
# 1. 读取所有岗位
# ---------------------------------------------------------

jobs = load_jobs("jobs.json")

print("========== 全部岗位 ==========")

for job in jobs:
    print(job)


# ---------------------------------------------------------
# 2. 筛选杭州岗位
# ---------------------------------------------------------

hangzhou_jobs = filter_by_city(jobs, "杭州")

print("\n========== 杭州岗位 ==========")

for job in hangzhou_jobs:
    print(
        job["company"],
        job["position"]
    )


# ---------------------------------------------------------
# 3. 统计技能出现次数
# ---------------------------------------------------------

skill_count = count_skills(jobs)

print("\n========== 技能出现次数 ==========")

# .items() 可以同时得到：
# skill：技能名称
# count：出现次数
for skill, count in skill_count.items():
    print(f"{skill}: {count}")


# ---------------------------------------------------------
# 4. 输出每个岗位与我的匹配率
# ---------------------------------------------------------

print("\n========== 岗位匹配率 ==========")

for job in jobs:

    rate = calculate_match_rate(job, my_skills)

    # rate 原本可能是 0.5
    # * 100 后变成 50
    #
    # :.0f 表示：
    # 不显示小数位
    #
    # 最后加 %，于是显示 50%
    print(
        f'{job["company"]} - '
        f'{job["position"]}：'
        f'{rate * 100:.0f}%'
    )


# ---------------------------------------------------------
# 5. 找到最匹配的岗位
# ---------------------------------------------------------

best_job, best_rate = find_best_job(
    jobs,
    my_skills
)

print("\n========== 最适合你的岗位 ==========")

# best_job 有可能是 None，
# 所以先判断一下
if best_job is not None:

    print("公司：", best_job["company"])
    print("岗位：", best_job["position"])
    print("城市：", best_job["city"])
    print("技能要求：", best_job["skills"])

    print(
        f"匹配率：{best_rate * 100:.0f}%"
    )

else:
    print("没有找到合适的岗位")