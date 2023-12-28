import torch
## Broadcasting
    # By now, you know how to perform elementwise binary operations on two tensors of the same shape.
    # Under certain conditions, even when shapes differ, we can still perform elementwise binary operations
    # by invoking the broadcasting mechanism. Broadcasting works according to the following two-step procedure: 
    # (i) expand one or both arrays by copying elements along axes with length 1 so that after this transformation, the two tensors have the same shape; 
    # (ii) perform an elementwise operation on the resulting arrays.


a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)

##Saving Memory
    # Running operations can cause new memory to be allocated to host results. 
    # For example, if we write Y = X + Y, we dereference the tensor that Y used to point to
    #  and instead point Y at the newly allocated memory. We can demonstrate this issue with Python’s id() function,
    #  which gives us the exact address of the referenced object in memory. 
    # Note that after we run Y = Y + X, id(Y) points to a different location. 
    # That is because Python first evaluates Y + X, allocating new memory for the result and then points Y to this new location in memory.

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
# On the other hand, X += Y modifies the existing tensor in-place, preserving the memory address of the original 
# tensor (X). The id() function in Python returns the identity of an object, which is essentially its memory address, 
# and that's why id(X) remains the same after the in-place addition.

##Conversion to other Python Objects

#Converting to a NumPy tensor (ndarray), or vice versa, is easy. The torch tensor and NumPy array will share their underlying memory, and changing one through an in-place operation will also change the other.
A = X.numpy()
A = X.numpy()
B = torch.from_numpy(A)
print(type(A), type(B))

# To convert a size-1 tensor to a Python scalar, we can invoke the item function or Python’s built-in functions.
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))

