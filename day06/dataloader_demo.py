import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# 1. 创建输入特征
X = torch.tensor([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [4.0, 3.0],
    [5.0, 6.0],
    [6.0, 5.0]
])


# 2. 创建标签
y = torch.tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0]
])


# 3. 把 X 和 y 组合成 Dataset
dataset = TensorDataset(
    X,
    y
)


# 4. DataLoader 按批次读取数据
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True   #翻译作洗牌，打乱样本原有的数据
)


# 5. 查看每个 batch
for batch_X, batch_y in loader:

    print("batch_X:")
    print(batch_X)

    print("batch_y:")
    print(batch_y)

    print("------")