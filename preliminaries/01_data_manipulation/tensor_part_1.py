import torch

# Creating a tensor using torch.arange
x = torch.arange(12, dtype=torch.float32)
print(f"Creating tensors with arange:\n{x}\n")

# Number of elements in the tensor
print(f"Number of elements in a tensor: {x.numel()}\n")

# Shape of the tensor
print(f"Shape of a tensor: {x.shape}\n")

# Reshaping the tensor
X = x.reshape(3, 4)
print(f"Reshaping our tensor:\n{X}\n")

# Tensors with all elements set to zeros and a shape of (2,3,4)
print(f"Tensors with all elements set to zero's and a shape of (2,3,4):\n{torch.zeros((2, 3, 4))}\n")

# Tensors with all elements set to ones and a shape of (2,3,4)
print(f"Tensors with all elements set to one's and a shape of (2,3,4):\n{torch.ones((2, 3, 4))}\n")

# Elements drawn from a standard Gaussian (normal) distribution
print(f"Elements drawn from a standard Gaussian distribution (mean 0, std 1):\n{torch.randn(3, 4)}\n")

# Creating a tensor with manual values
print(f"Manual values:\n{torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])}\n")

## Indexing and Slicing
# Last row of our tensor
print(f"Last row of our Tensor:\n{X[-1]}\n")
# 2nd and 3rd row of the tensor
print(f"2nd and 3rd row of tensor:\n{X[1:3]}")

# Writing elements
X[1, 2] = 17
print(f"Writing elements:\n{X}")

# Assigning multiple elements with the same value
X[:2, :] = 12
print(f"\nAssigning multiple elements with the same value:\n{X}")

## Operations
# Exponential function applied element-wise
torch.exp(x)

# Element-wise operations
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

# Concatenating tensors along specified dimensions
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

# Element-wise equality check
print(X == Y)

# Summation of all elements
print(f"Summation of all elements: {X.sum()}")
