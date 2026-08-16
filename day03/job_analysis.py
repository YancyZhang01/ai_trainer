from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

FILE_PATH = DATA_DIR / "jobs_clean.csv"



# CSV → DataFrame

df = pd.read_csv(FILE_PATH)

df["skills"] = df["skills"].fillna("")
# value_counts()
#
# 统计city这一列中
# 每一种城市出现多少次。

city_counts = df["city"].value_counts()

print(city_counts)
skill_count = {}


# 遍历skills这一列
#
# 每一次skills_text
# 类似：
#
# "python,rag,llm"

for skills_text in df["skills"]:


    # split(",")
    #
    # "python,rag,llm"
    #
    # ↓
    #
    # ["python", "rag", "llm"]

    skills = skills_text.split(",")


    for skill in skills:

        # 如果字典中已经有这个技能

        if skill in skill_count:

            skill_count[skill] += 1

        else:

            # 第一次出现

            skill_count[skill] = 1


print(skill_count)

sorted_skills = sorted(
    skill_count.items(),
    key=lambda x: x[1],
    reverse=True
)

print(skill_count)