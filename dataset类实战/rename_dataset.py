import os

root_dir = "../dataset/train"
target_dir = "bees_image"
image_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split("_")[0]
out_dir = "bees_label"
# 确保输出目录存在
out_dir_path = os.path.join(root_dir, out_dir)
if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)  # 创建目录

for i in image_path:
    file_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), "w") as f:
        f.write(label)
