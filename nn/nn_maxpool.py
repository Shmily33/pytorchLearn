import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]
                      ], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))  # shape()第一个参数为-1，就可以让他自己计算batch_size
print(input.shape)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


net = Net()
output = net(input)
print(output)

dataset = torchvision.datasets.CIFAR10(root='../torchvision_dataloader/dataset', train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)
writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input_maxpool", imgs, step)
    outputs = net(imgs)  # 最大池化不会改变channel，所以不用reshape
    writer.add_images("output_maxpool", outputs, step)
    step += 1

writer.close()
