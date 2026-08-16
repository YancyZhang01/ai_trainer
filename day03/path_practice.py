from pathlib import Path


# 当前 py 文件所在目录
BASE_DIR = Path(__file__).resolve().parent


# data 文件夹路径
DATA_DIR = BASE_DIR / "data"


print("当前文件夹:")
print(BASE_DIR)


print("\ndata路径:")
print(DATA_DIR)


# 判断 data 是否存在
if DATA_DIR.exists():

    print("data文件夹已经存在")

else:

    print("data不存在，正在创建")

    DATA_DIR.mkdir()

    print("创建完成")