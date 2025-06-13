import os
import random
import datetime
from PIL import Image, ImageDraw, ImageFont

# 参数配置
background_path = "./image/backgroud.png"  # 背景图文件名
output_dir = "generated_dataset3"
font_path = "/home/bz/workspace/ultralytics/generate/Perfect DOS VGA 437.ttf"  # 宋体字体路径，按需修改
image_count = 20000
image_size = (1580, 742)  # 背景图分辨率
digit_font_size = 60
time_font_size = 50
color_red = (255, 0, 0)

# 创建输出目录
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/labels", exist_ok=True)

# 加载背景图和字体
background = Image.open(background_path).convert("RGB")
digit_font = ImageFont.truetype(font_path, digit_font_size)
time_font = ImageFont.truetype(font_path, time_font_size)

def draw_text_with_boxes(draw, text, font, start_pos, spacing, fill_color=color_red):
    boxes = []
    x, y = start_pos
    for char in text:
        bbox = draw.textbbox((x, y), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((x, y), char, font=font, fill=fill_color)
        cx = (x + w / 2) / image_size[0]
        cy = (y + h / 2) / image_size[1]
        if char in "0123456789":
            class_id = int(char)
            boxes.append(f"{class_id} {cx:.6f} {cy:.6f} {w / image_size[0]:.6f} {h / image_size[1]:.6f}")
        x += w + spacing
    return boxes

for i in range(image_count):
    img = background.copy()
    draw = ImageDraw.Draw(img)

    # 左上角6位数字
    left_digits = ''.join(random.choices("0123456789", k=6))
    left_boxes = draw_text_with_boxes(draw, left_digits, digit_font, (20, 20), spacing=-4)

    # 右上角时间戳
    now = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 1000000))
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    bbox = draw.textbbox((0, 0), time_str, font=time_font)
    time_text_width = bbox[2] - bbox[0]
    start_x = image_size[0] - time_text_width - 150
    time_boxes = draw_text_with_boxes(draw, time_str, time_font, (start_x, 20), spacing=-4)

    # 保存图像和标签
    img_path = f"{output_dir}/images/{i:05}.jpg"
    label_path = f"{output_dir}/labels/{i:05}.txt"
    img.save(img_path)
    with open(label_path, 'w') as f:
        f.write("\n".join(left_boxes + time_boxes))

    if (i + 1) % 100 == 0:
        print(f"已生成 {i+1} / {image_count} 张")

print("✅ 数据集生成完毕，共生成 10000 张图像和标注。")
