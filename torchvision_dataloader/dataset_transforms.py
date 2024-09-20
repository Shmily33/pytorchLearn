import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transforms,download=True)

print(test_set[0])
# img, target = train_set[0]
# img.show()
writer = SummaryWriter(log_dir='./logs')
for i in range(10):
    image, target = train_set[i]
    writer.add_image('test_set', image, i)

writer.close()