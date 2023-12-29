import torch

"""
Scalars

(Scalars are implemented as tensors that contain only one element.) Below, we assign two scalars
and perform the familiar addition, multiplication, division, and exponentiation operations.

"""
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x**y)

""" 
Vectors are implemented as  1st -order tensors. In general, such tensors can have arbitrary lengths, 
subject to memory limitations. Caution: in Python, as in most programming languages, 
vector indices start at  0 , also known as zero-based indexing, whereas in linear algebra 
subscripts begin at  1  (one-based indexing).

"""
x = torch.arange(3)
print(x)

print(x[2])

print(len(x))

print(x.shape)

""" 
Matrices

"""

A = torch.arange(6).reshape(3, 2)
print(A)

print(A.T)

A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(A == A.T)

""" 
Tensors 

Tensors will become more important when we start working with images. Each image arrives as a  
3rd -order tensor with axes corresponding to the height, width, and channel. At each spatial location, 
the intensities of each color (red, green, and blue) are stacked along the channel. Furthermore, 
a collection of images is represented in code by a  4th -order tensor, where distinct images are 
indexed along the first axis. Higher-order tensors are constructed, as were vectors and matrices, 
by growing the number of shape components.

"""
print(torch.arange(24).reshape(2, 3, 4))

## Tensor Arithmetic
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
print(A, A + B)

print(A * B) #Element wise product of two matrices.

""" 
Adding or multiplying a scalar and a tensor produces a result with the same shape 
as the original tensor. Here, each element of the tensor is added to (or multiplied by) the scalar.
"""
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

""" 
Calculating Sum of Tensor 

"""
x = torch.arange(3, dtype=torch.float32)
print(x, x.sum())

print(A.shape(),A.sum())

"""
By default, invoking the sum function reduces a tensor along all of its axes, eventually producing a scalar. 
Our libraries also allow us to [specify the axes along which the tensor should be reduced.] To sum over 
all elements along the rows (axis 0), we specify axis=0 in sum. Since the input matrix reduces along 
axis 0{row summation} to generate the output vector, this axis is missing from the shape of the output. 
"""

print(A.shape, A.sum(axis=0).shape)
print(A.shape, A.sum(axis=1).shape)

print(A.sum(axis=[0, 1]) == A.sum())  # Same as A.sum()

print(A.mean(), A.sum() / A.numel())

print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

""" 
Non Reduction sum

Sometimes it can be useful to [keep the number of axes unchanged] when invoking the 
function for calculating the sum or mean. This matters when we want to use the broadcast mechanism.

"""
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A, sum_A.shape)

""" 
If we want to calculate [the cumulative sum of elements of A along some axis], say axis=0 (row by row), 
we can call the cumsum function. By design, this function does not reduce the input tensor along any axis.

"""
print(A / sum_A)
"""
If we want to calculate [the cumulative sum of elements of A along some axis], say axis=0 (row by row), 
we can call the cumsum function. By design, this function does not reduce the input tensor along any axis.
"""
print(A.cumsum(axis=0))

"""Dot Products"""

y = torch.ones(3, dtype = torch.float32)
print(x, y, torch.dot(x, y))

