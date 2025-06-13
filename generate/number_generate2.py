import os
import random
import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob

background_dir = "./image/backgrounds"
output_dir = "generated_dataset_dynamic_bg_centered"
font_path = "/home/bz/workspace/ultralytics/generate/Perfect DOS VGA 437.ttf"
image_count = 20000

digit_font_size = 90   # 加大数字字体
time_font_size = 90    # 加大时间字体
color_red = (255, 0, 0)

os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/labels", exist_ok=True)

background_paths = glob.glob(os.path.join(background_dir, "*.*"))
assert background_paths, f"目录 {background_dir} 为空或不存在背景图"

digit_font = ImageFont.truetype(font_path, digit_font_size)
time_font = ImageFont.truetype(font_path, time_font_size)

def draw_text_with_boxes_centered(draw, text, font, box, image_size, fill_color=color_red):
    """
    在给定box区域内水平居中绘制文本，返回YOLO格式的标注列表
    box = (x0, y0, width, height)
    """
    x0, y0, box_w, box_h = box

    # 先计算文本总宽度（字符宽度+spacing），这里spacing设0避免重叠
    spacing = -4
    total_w = 0
    char_sizes = []
    for char in text:
        bbox = draw.textbbox((0,0), char, font=font)
        w = bbox[2] - bbox[0]
        char_sizes.append(w)
        total_w += w + spacing
    total_w -= spacing  # 最后一个字符不用加spacing

    # 计算起始x，使文本整体水平居中在box内
    start_x = x0 + (box_w - total_w) / 2
    # 纵向居中，字体高度
    bbox_h = font.getbbox(text)[3] - font.getbbox(text)[1]
    start_y = y0 + (box_h - bbox_h) / 2

    boxes = []
    x = start_x
    for idx, char in enumerate(text):
        w = char_sizes[idx]
        bbox_char = draw.textbbox((x, start_y), char, font=font)
        h = bbox_char[3] - bbox_char[1]

        # 绘制字符
        draw.text((x, start_y), char, font=font, fill=fill_color)

        # 计算字符中心
        cx = (x + w / 2) / image_size[0]
        cy = (start_y + h / 2) / image_size[1]

        if char in "0123456789":
            class_id = int(char)
            boxes.append(f"{class_id} {cx:.6f} {cy:.6f} {w / image_size[0]:.6f} {h / image_size[1]:.6f}")

        x += w + spacing

    return boxes

def add_noise_and_blur(img):
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    return img

for i in range(image_count):
    bg_path = random.choice(background_paths)
    background = Image.open(bg_path).convert("RGB")
    image_size = background.size  # (宽, 高)

    img = background.copy()
    img = add_noise_and_blur(img)
    draw = ImageDraw.Draw(img)

    # 定义左上角数字显示区域框，宽高可调，确保数字能显示完整且居中
    left_box = (20, 20, 500, digit_font_size + 20)

    left_digits = ''.join(random.choices("0123456789", k=6))
    left_boxes = draw_text_with_boxes_centered(draw, left_digits, digit_font, left_box, image_size)

    # 定义右上角时间戳区域框
    time_box_width = 700
    time_box = (image_size[0] - time_box_width - 150, 20, time_box_width, time_font_size + 20)

    now = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 1000000))
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    time_boxes = draw_text_with_boxes_centered(draw, time_str, time_font, time_box, image_size)

    img_path = f"{output_dir}/images/{i:05}.jpg"
    label_path = f"{output_dir}/labels/{i:05}.txt"
    img.save(img_path)
    with open(label_path, 'w') as f:
        f.write("\n".join(left_boxes + time_boxes))

    if (i + 1) % 100 == 0:
        print(f"已生成 {i+1} / {image_count} 张")

print(f"✅ 数据集生成完毕，共生成 {image_count} 张图像和标注。")
