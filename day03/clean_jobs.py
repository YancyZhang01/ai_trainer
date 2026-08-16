from pathlib import Path
import pandas as pd


# =========================================================
# 1. 构造文件路径
# =========================================================

# __file__
# 当前Python文件的位置
#
# Path(...)
# 转换成Path对象
#
# resolve()
# 解析为绝对路径
#
# parent
# 取得当前.py文件所在文件夹

BASE_DIR = Path(__file__).resolve().parent


# / 在Path对象中表示“路径拼接”
#
# day03
# ↓
# day03/data

DATA_DIR = BASE_DIR / "data"


# day03/data/jobs_raw.csv

INPUT_FILE = DATA_DIR / "jobs_raw.csv"


# =========================================================
# 2. 定义技能清洗函数
# =========================================================

def clean_skills(text):

    # split(",")
    # 字符串 → 列表

    skills = text.split(",")

    clean_result = []

    for skill in skills:

        # 删除左右空格 + 转小写

        skill = skill.strip().lower()

        if skill:
            clean_result.append(skill)

    # 列表 → 字符串

    return ",".join(clean_result)


# =========================================================
# 3. 读取CSV
# =========================================================

# pd.read_csv()
# CSV → DataFrame

df = pd.read_csv(INPUT_FILE)


# =========================================================
# 4. 初步检查数据
# =========================================================

# head()
# 查看前5行

print("===== 前5行 =====")
print(df.head())


# shape
# (行数, 列数)

print("\n===== 数据规模 =====")
print(df.shape)


# isna()
# 判断是否缺失
#
# sum()
# 统计每列有多少缺失值

print("\n===== 缺失值 =====")
print(df.isna().sum())


# =========================================================
# 5. 删除重复行
# =========================================================

before_count = len(df)

df = df.drop_duplicates()

after_count = len(df)

print("\n清洗前行数：", before_count)
print("去重后行数：", after_count)


# =========================================================
# 6. 处理缺失值
# =========================================================

# fillna()
# 将city缺失值填成“未知”

df["city"] = df["city"].fillna("未知")


# skills缺失时填成空字符串

df["skills"] = df["skills"].fillna("")


# =========================================================
# 7. 清洗skills
# =========================================================

# apply(clean_skills)
#
# 对skills列里的每一个值
# 调用一次clean_skills函数

df["skills"] = df["skills"].apply(clean_skills)


# =========================================================
# 8. 保存结果
# =========================================================

OUTPUT_CSV = DATA_DIR / "jobs_clean.csv"

OUTPUT_JSON = DATA_DIR / "jobs_clean.json"


# DataFrame → CSV

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# DataFrame → JSON

df.to_json(
    OUTPUT_JSON,
    orient="records",
    force_ascii=False,
    indent=4
)


print("\n数据清洗完成！")

print("CSV保存位置：", OUTPUT_CSV)

print("JSON保存位置：", OUTPUT_JSON)