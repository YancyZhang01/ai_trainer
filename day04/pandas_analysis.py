import pandas as pd


data = {
    "city": [
        "杭州",
        "上海",
        "杭州",
        "上海"
    ],

    "salary": [
        200,
        300,
        250,
        350
    ]
}


df = pd.DataFrame(data)


# groupby("city")
#
# 作用：
# 按 city 这一列进行分组。
#
# 可以理解为：
#
# 杭州组：
# 200
# 250
#
# 上海组：
# 300
# 350


# ["salary"]
#
# 分组以后只看 salary 列。
#
# .mean()
#
# 计算每组平均值。

result = (
    df
    .groupby("city")["salary"]
    .mean()
)


print(result)