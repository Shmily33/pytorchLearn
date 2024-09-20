import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../torchvision_dataloader/dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
print(net)
writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = net(imgs)
    # print(imgs.shape)
    # print(output.shape)

    writer.add_images("input_conv2d", imgs, step)  # torch.Size([64, 3, 32, 32])
    # torch.Size([64, 6, 32, 32]) -> [xx, 3, 32, 32]
    # writer.add_images("output", output, step) # 直接运行会报错，这个通道是6了，只能显示3，所以要reshape
    output = torch.reshape(output, [-1, 3, 32, 32])  # 不知道第一个参数即batch_size,可以先写-1 会自己计算
    # print(output.shape)
    writer.add_images("output_conv2d", output, step)  # 直接运行会报错，这个通道是6了，只能显示3，所以要reshape
    step += 1

writer.close()
