from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

# print(cv2.__version__)
writer = SummaryWriter("logs")

image_path = "../dataset/train/ants_image/0013035.jpg"
# cv_img = cv2.imread(image_path)
# print(cv_img)
image = Image.open(image_path)
print(image)

# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(image)
writer.add_image("ToTensor", tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([-1, -2, -3], [3, 2, 1])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 3)

# Resize
print(image.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(image)
# img_resize PIL -> toTensor ->img_resize tensor
img_resize = tensor_trans(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize.size)

# Compose
trans_resize_2 = transforms.Resize(600)  # smaller edge of the image will be matched to this number.
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, transforms.ToTensor()])
img_resize_2 = trans_compose(image)
# Convert tensor to PIL image to check size
img_resize_2_pil = transforms.ToPILImage()(img_resize_2)
print(img_resize_2_pil.size)  # Print composed image size
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random, transforms.ToTensor()])
for i in range(10):
    img_crop = trans_compose_2(image)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
