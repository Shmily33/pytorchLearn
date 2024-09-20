from torch.utils.data import Dataset
from PIL import Image
import os


# help(Dataset)
class MyData(Dataset):

    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(os.path.join(self.root_dir, self.img_dir))
        self.label_path = os.listdir(os.path.join(self.root_dir, self.label_dir))

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        label_name = self.label_path[idx]
        image = Image.open(os.path.join(self.root_dir, self.img_dir, image_name))
        with open(os.path.join(self.root_dir, self.label_dir, label_name), 'r') as f:
            label = f.readline()

        return image, label

    def __len__(self):
        return len(self.image_path)


root_dir = "../dataset/train"
img_dir = "ants_image"
label_dir = "ants_label"
ants_dataset = MyData(root_dir, img_dir, label_dir)
img_dir = "bees_image"
label_dir = "bees_label"
bees_dataset = MyData(root_dir, img_dir, label_dir)

train_dataset = ants_dataset + bees_dataset

# 测试
image, label = bees_dataset[1]
print(image)
print(label)
image.show()

