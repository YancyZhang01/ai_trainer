import pandas as pd

data = {

    "company": ["A", "B", "C"],

    "city": [
        "杭州",
        "上海",
        "杭州"
    ],

    "salary": [
        200,
        300,
        250
    ]
}


 #字典dataframe，可以理解为Python里的Excel表格。
df = pd.DataFrame(data)  

print(df)

# 读取csv文件
df = pd.read_csv("day03/data/jobs.csv")

print("CSV读取结果:")
print(df)

# 查看前5行
print(df.head())

#看表结构
print(df.shape)
print(df.columns)
print(df.info())

#选择某一列
hangzhou_jobs = df[df["city"] == "杭州"]

print(hangzhou_jobs)

print(df["city"].value_counts())         #value_counts

print(df.isna().sum())         #缺失值

df = df.dropna()             #删除缺失值