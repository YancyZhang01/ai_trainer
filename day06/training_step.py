import torch
from torch import nn

# 一个非常简单的模型
model = nn.Linear(
    2,
    1
)

# 二分类 Loss
criterion = nn.BCEWithLogitsLoss()


# 优化器，随机梯度下降根据梯度更新模型参数的算法。
optimizer = torch.optim.SGD(  
    model.parameters(),
    lr=0.01 #learning rate学习率，迈多大的步子
)

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
y = torch.tensor([
    [0.0],
    [1.0]
])

# 1. Forward，把数据 X 输入模型，让模型先预测一次。
logits = model(X)


# 2. 计算 Loss
loss = criterion(
    logits,
    y
)

# 3. 清空旧梯度，pytorch 默认梯度会累加，所以反向传播前需要清空上一轮的梯度
optimizer.zero_grad()


# 4. 反向传播，根据 Loss，反向计算模型里的每个参数应该往哪个方向调整。
loss.backward()


# 5. 根据反向传播结果，更新模型参数
optimizer.step()


print("Loss:")
print(loss.item())