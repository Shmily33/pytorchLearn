import torch
import torchvision.transforms
from PIL import Image

image_path = "./images/img.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
image = torchvision.transforms.Compose([torchvision.transforms.Resize([32, 32]),
                                        torchvision.transforms.ToTensor()])(image)
print(image.shape)
model = torch.load("net_1.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))