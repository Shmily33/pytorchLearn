import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

# output = torch.reshape(input, (-1, 1, 2, 2))
# print(output.shape)

dataset = torchvision.datasets.CIFAR10(root='../torchvision_dataloader/dataset', train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = ReLU()  # ReLU(x)=max(0,x)
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        # x = self.relu1(x)
        x = self.sigmoid1(x)
        return x


net = Net()
# output = net(input)
# print(output)

writer = SummaryWriter("./logs_relu_sigmoid")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input_sigmoid", imgs, step)
    outputs = net(imgs)
    writer.add_images("output_sigmoid", outputs, step)
    step += 1

writer.close()
