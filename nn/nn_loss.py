import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)
loss = L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)

lose_mse = nn.MSELoss()
result_mse = lose_mse(inputs, targets)
print(result_mse)
x = torch.tensor([1, 2, 3]).reshape(1, 3)
print(x.numpy())