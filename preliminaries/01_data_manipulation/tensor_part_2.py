import torch

## Broadcasting
# Performing elementwise binary operations on tensors of the same shape is a fundamental concept.
# In cases where shapes differ, we can leverage the broadcasting mechanism, which works through the following steps:
# (i) Expand one or both arrays by copying elements along axes with length 1, ensuring the two tensors have the same shape.
# (ii) Perform an elementwise operation on the resulting arrays.

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)

## Saving Memory
# Running operations can allocate new memory for results. For instance, if we write Y = X + Y,
# we dereference the tensor that Y used to point to, and instead, point Y at the newly allocated memory.
# This is demonstrated using Python’s id() function, which provides the exact address of the referenced object in memory.
# After running Y = Y + X, id(Y) points to a different location, as Python first evaluates Y + X, allocating new memory for the result.

Y = torch.randn(3, 4)
X = torch.randn(3, 4)
before = id(Y)
Y = Y + X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)

# In summary, Y = Y + X involves the creation of a new tensor, leading to a change in the memory address referenced by Y.
# On the other hand, X += Y modifies the existing tensor in-place, preserving the memory address of the original tensor (X).
# The id() function in Python returns the identity of an object, which is essentially its memory address.

## Conversion to other Python Objects

# Converting between a torch tensor and NumPy array is straightforward.
# The torch tensor and NumPy array will share their underlying memory, so changes to one will affect the other.

A = X.numpy()
B = torch.from_numpy(A)
print(type(A), type(B))

# To convert a size-1 tensor to a Python scalar, we can use the item function or Python’s built-in functions.
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
