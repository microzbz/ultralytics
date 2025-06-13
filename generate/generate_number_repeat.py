import os
import random
import shutil
from collections import defaultdict

# 原始目录
image_dir = "/home/bz/workspace/ultralytics/generate/generated_dataset/images"
label_dir = "/home/bz/workspace/ultralytics/generate/generated_dataset/labels"

# 输出目录
output_image_dir = "generated_dataset_repeat_image"
output_label_dir = "generated_dataset_repeat_label"

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 支持的图像格式
image_exts = [".jpg", ".png"]

# 获取所有图像文件名（不含扩展名）
all_images = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_exts]
image_names = [os.path.splitext(f)[0] for f in all_images]

# 随机选出 1000 张图像用于重复
selected = random.sample(image_names, min(1000, len(image_names)))

# 分配每张图像的重复次数，目标总数约 10000
total_target = 10000
repeat_counts = defaultdict(int)

# 初始都 1 次
for name in selected:
    repeat_counts[name] = 1

# 继续分配重复次数直到接近目标数量
current_total = sum(repeat_counts.values())
while current_total < total_target:
    name = random.choice(selected)
    repeat_counts[name] += 1
    current_total += 1

# 复制图像和标签文件
counter = 0
for name, repeat in repeat_counts.items():
    image_path = None
    for ext in image_exts:
        if os.path.exists(os.path.join(image_dir, name + ext)):
            image_path = os.path.join(image_dir, name + ext)
            break

    label_path = os.path.join(label_dir, name + ".txt")
    if not image_path or not os.path.exists(label_path):
        print(f"⚠️ 跳过缺失文件：{name}")
        continue

    for i in range(repeat):
        new_name = f"{name}_aug_{i}_{counter}"
        new_image_ext = os.path.splitext(image_path)[1].lower()
        new_image_path = os.path.join(output_image_dir, new_name + new_image_ext)
        new_label_path = os.path.join(output_label_dir, new_name + ".txt")

        shutil.copyfile(image_path, new_image_path)
        shutil.copyfile(label_path, new_label_path)
        counter += 1

print(f"✅ 完成：共生成 {counter} 张图像及标签，保存在 {output_image_dir}/ 与 {output_label_dir}/ 中")
