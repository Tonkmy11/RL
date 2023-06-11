import torch
import numpy as np
from torch.autograd import Variable

data = [1, 2], [3, 4]
"""
tensor = torch.FloatTensor(data)
data = np.array(data)
print('\n', data)
print('\n', data.dot(data))
print('\n', torch.mm(tensor, tensor))
"""

tensor = torch.FloatTensor(data)
tensor.requires_grad = True

t_out = torch.mean(tensor*tensor)
t_out.backward()

# t_out = 1/4*sum(tensor^2)
# d(t_out)/d(tensor) = 1/4*2*tensor = tensor/2

print('\n', tensor.grad)
print('\n', tensor.data)
print('\n', tensor.data.numpy())
