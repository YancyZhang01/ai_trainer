import torch

from torch import nn


class SimpleNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(
            2,
            4
        )

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(
            4,
            1
        )

# #执行顺序x
# ↓
# Linear
# ↓
# ReLU
# ↓
# Linear
# ↓
# output
    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        return x


model = SimpleNetwork()


sample = torch.tensor([
    [1.0, 2.0]
])

#！！！！！！默认写法！！！
output = model(sample)  #PyTorch 会自动调用：forward（）


print(model)

print("\noutput:")
print(output)