"""
对应model_save 此py文件是加载model_save 保存的网络模型
"""
import torch
import torchvision
from torch import nn
from model_save import *

# 方式1的加载
model1 = torch.load("vgg_method1.pth")
# print(model1)
# 方式2的加载
model2 = torch.load("vgg_method2.pth")
print(model2)  # 就只有参数
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(model2)
print(vgg16)

# 陷阱--自定义网络时如
net = torch.load("net.pth")
print(net)
"""
直接加载的话会报错：AttributeError: Can't get attribute 'Net'
                on <module '__main__' from 'E:\\桌面\\pytorchLearn\\model\\model_load.py'>
因为不确定自定义的网络模型
解决方式1：
        将net的类定义复制到此文件中
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
            def forward(self, x):
                x = self.conv1(x)
                return x
解决方式2：
        引入定义Net类的模块文件
        如：from model_save import *
"""
