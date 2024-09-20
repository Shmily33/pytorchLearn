import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1--不仅保存了网络模型的结构还保存了网络模型的参数(模型结构+模型参数)
torch.save(vgg16, "vgg_method1.pth")

# 保存方式2--保存参数为字典(模型参数) -> 官方推荐
torch.save(vgg16.state_dict(), "vgg_method2.pth")


# 陷阱--自定义网络时如
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()

torch.save(net, "net_save.pth")
