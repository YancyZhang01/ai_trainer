import torch

from torch import nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


# =========================================================
# 1. 准备数据
# =========================================================

X = torch.tensor([
    [0.5, 1.0],
    [1.0, 0.8],
    [1.5, 1.2],
    [2.0, 1.5],
    [3.5, 4.0],
    [4.0, 3.5],
    [4.5, 4.2],
    [5.0, 4.8],
    [5.5, 5.2],
    [6.0, 5.5]
], dtype=torch.float32)


y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0]
], dtype=torch.float32)


# =========================================================
# 2. Dataset + DataLoader
# =========================================================

dataset = TensorDataset(
    X,
    y
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# =========================================================
# 3. 定义模型
# =========================================================

class BinaryClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(      #nn.Sequential 把多个网络层按照顺序组合起来。输入首先经过 Linear(2,8)，从 2 个特征变成 8 个特征；然后经过 ReLU；最后经过 Linear(8,1)，把 8 个特征变成 1 个输出。这样在 forward() 中直接调用 self.network(x) 就可以让数据依次经过所有层
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )


    def forward(self, x):

        return self.network(x)


model = BinaryClassifier()


# =========================================================
# 4. Loss + Optimizer
# =========================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(  #优化器，后面调用
    model.parameters(),
    lr=0.05
)


# =========================================================
# 5. Training Loop
# =========================================================

epochs = 100


for epoch in range(epochs):

    total_loss = 0.0


    for batch_X, batch_y in loader:

        # Forward，其中logits是最终原始分数。
        logits = model(batch_X)

        # Loss
        loss = criterion(
            logits,
            batch_y
        )

        # 清空梯度
        optimizer.zero_grad()

        # Backward
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()


    # 每10轮输出一次
    if (epoch + 1) % 10 == 0:

        avg_loss = (
            total_loss
            / len(loader)   #一个 Epoch 中有多少个 batch。
        )

        print(
            f"Epoch {epoch + 1:3d} "
            f"Loss: {avg_loss:.4f}"
        )


# =========================================================
# 6. Evaluation
# =========================================================

model.eval()


with torch.no_grad():

    logits = model(X)

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= 0.5
    ).float()


    accuracy = (
        predictions == y
    ).float().mean()


print("\n===== Prediction =====")

print(
    "Probabilities:"
)

print(
    probabilities.squeeze()
)


print(
    "\nPredictions:"
)

print(
    predictions.squeeze()
)


print(
    "\nLabels:"
)

print(
    y.squeeze()
)


print(
    f"\nAccuracy: "
    f"{accuracy.item():.3f}"
)