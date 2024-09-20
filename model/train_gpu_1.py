from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
import torch
import torchvision

# 利用dataset准备数据集
train_data = torchvision.datasets.CIFAR10("../torchvision_dataloader/dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("../torchvision_dataloader/dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
# 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Train data size: {}".format(train_data_size))
print("Test data size: {}".format(test_data_size))
# 利用dataloader加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)
# 创建网络模型
net = Net()
if torch.cuda.is_available():
    net = net.cuda()  # gpu
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # gpu
# 学习速率
learning_rate = 1e-2
# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
# 设置网络训练的一些参数
total_train_step = 0  # 总的训练次数
total_test_step = 0  # 总的测试次数
epoch = 10  # 训练轮数

writer = SummaryWriter("./logs")

for i in range(epoch):  # 训练10抡、
    print("--------第{}次训练开始：--------".format(i + 1))
    # 训练
    net.train()
    for data in train_data_loader:
        images, targets = data
        if torch.cuda.is_available():
            images, targets = images.cuda(), targets.cuda()  # gpu
        outputs = net(images)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{},loss:{}".format(total_train_step, loss.item()))
            # torch.save(net, "net_gpu_1.pth")
            writer.add_scalar("train_loss_gpu", loss.item(), total_train_step)

    # 测试
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, targets = data
            if torch.cuda.is_available():
                images, targets = images.cuda(), targets.cuda()  # gpu
            outputs = net(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()  # argmax(1) 一行的最大数 例子accuracy=[false,true]
            total_accuracy += accuracy
        print("整体测试集上的loss:{}".format(total_test_loss))
        writer.add_scalar("test_loss_gpu", total_test_loss, total_test_step)
        print("整体测试集上的accuracy:{}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_accuracy_gpu", total_accuracy / test_data_size, total_test_step)
        total_test_step += 1

writer.close()
