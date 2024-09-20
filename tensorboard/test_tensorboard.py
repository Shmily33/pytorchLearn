from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter('logs')
# dataset/train/ants_image/0013035.jpg
image_path = "../dataset/train/ants_image/0013035.jpg"
image = Image.open(image_path)
image_array = np.array(image)
writer.add_image("test", image_array, 3, dataformats='HWC')
writer.add_image("train", image_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar('y=2x', i * 3, i)  # 标题 y对应值 x对应值
writer.close()
