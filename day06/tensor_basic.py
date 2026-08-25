import torch


# 1. 从 Python list 创建 Tensor
a = torch.tensor([1, 2, 3])

print("a:")
print(a)

print("shape:", a.shape)
print("dtype:", a.dtype)


# 2. 创建二维 Tensor
matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("\nmatrix:")
print(matrix)

print("shape:", matrix.shape)
print("ndim:", matrix.ndim)


# 3. 常见初始化方式
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 3))
random_tensor = torch.rand((2, 3))

print("\nzeros:")
print(zeros)

print("\nones:")
print(ones)

print("\nrandom:")
print(random_tensor)


# 4. Tensor 运算
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print("\nx + y:")
print(x + y)

print("\nx * 2:")
print(x * 2)

print("\ndot:")
print(torch.dot(x, y))