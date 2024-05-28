import torch

print(torch.__version__)

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float, device="cuda", requires_grad=True)

print(my_tensor)

print(my_tensor.device)

x = torch.rand(size=(3,3))
x = torch.zeros((4))
# x = torch.arange(start=0, end=5, step=2)
# print(x)

x = torch.empty(size=(1,5)).uniform_(0,1)
x = torch.empty(size=(1,5)).normal_(mean=0,std=1)

# tensor = torch.arange(4)
# print(tensor.bool())
# print(tensor.short())
# print(tensor.long())
# print(tensor.half())

import numpy as np
np_arr = np.zeros((5,5))
tensor = torch.from_numpy(np_arr)
np_arr_back = tensor.numpy()

# print(tensor)

# ===================================================== #
#         Tensor Math & Comparision Operations          #
# ===================================================== #

x = torch.tensor([1, 2, 5])
y = torch.tensor([4, 2, 6])
z = torch.empty(3)
torch.add(x , y, out  = z)

# inplace iterations
t = torch.zeros(3)
# t.add_(x)
t += x
print(t)

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
# print(x3)

# Matrix exponentiation
matrix_exp = torch.rand(5,5)
# print(matrix_exp.matrix_power(3))

z = torch.dot(x, y)
# print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # batch, m ,p

# Example of BroadCasting
x1 = torch.rand(5,5)
x2 = torch.rand((1,5))

z = x1 - x2

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values , indices = torch.min(x, dim=0)
# print(indices)

abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
# print(z)
z = torch.eq(x, y)
sorted_y , indices = torch.sort(y, dim = 0, descending=True)

z = torch.clamp(x, min=0, max=10)
a = torch.tensor([1,0,1,1,0], dtype=torch.bool)
# print(a)

# ============================================================== #
#                      Tensor Indexing                           #
# ============================================================== #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# print(x[0].shape) # x[0,:]
# print((x[:, 0].shape))
# print(x[2, 0:10]) # 0:10 --> [0,1,2, ....., 9]
#
# # Fancy indexing
# x = torch.arange(10)
# indices = [2, 5, 8]
# print(x[indices])
# x = torch.rand((3,5))
# rows = torch.tensor([1, 0])
# cols = torch.tensor([4,0])
# print(x[rows, cols])
#
# # More advanced indexing
# x = torch.arange(10)
# print(x[(x < 2) | (x > 8)])
# print(x[x.remainder(2) == 0])
#
# # useful operations
# print(torch.where(x>5, x , 2*x))
# print(torch.tensor([0,0,1,2,3,2,3,5]).unique())
# print(x.ndimension())
# print(x.numel())


# ===================================================== #
#                   Fancy Indexing                      #
# ===================================================== #

x = torch.arange(9)

x_3x3 = x.view(3, 3)
x_3x3 = x.reshape(3 , 3)
print(x_3x3)
y = x_3x3.t()
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape )
print(torch.cat((x1, x2), dim=1).shape )

z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10

z = x.squeeze(1)
print(z.shape)

import numpy as np

a = np.random.randn(3,4)
b = np.random.randn(4,5)

print(np.round(a@b , 2)) , print(" ")
print(np.matmul(a,b))

