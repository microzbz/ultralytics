import os
import random
import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob

# ====== 配置部分 ======
background_dir = "./image/backgrounds"
output_dir = "generated_dataset_special_088"
font_path = "/home/bz/workspace/ultralytics/generate/Perfect DOS VGA 437.ttf"
target_count = 5000
start_index = 26001

digit_font_size = 90
time_font_size = 90
color_red = (255, 0, 0)

# ====== 创建目录 ======
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

background_paths = glob.glob(os.path.join(background_dir, "*.*"))
assert background_paths, f"目录 {background_dir} 为空或不存在背景图"

digit_font = ImageFont.truetype(font_path, digit_font_size)
time_font = ImageFont.truetype(font_path, time_font_size)

# ====== 工具函数 ======
def draw_text_with_boxes_centered(draw, text, font, box, image_size, fill_color=color_red):
    x0, y0, box_w, box_h = box
    spacing = -4
    total_w = 0
    char_sizes = []

    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        char_sizes.append(w)
        total_w += w + spacing
    total_w -= spacing

    start_x = x0 + (box_w - total_w) / 2
    bbox_h = font.getbbox(text)[3] - font.getbbox(text)[1]
    start_y = y0 + (box_h - bbox_h) / 2

    boxes = []
    x = start_x
    for idx, char in enumerate(text):
        w = char_sizes[idx]
        bbox_char = draw.textbbox((x, start_y), char, font=font)
        h = bbox_char[3] - bbox_char[1]

        draw.text((x, start_y), char, font=font, fill=fill_color)

        cx = (x + w / 2) / image_size[0]
        cy = (start_y + h / 2) / image_size[1]

        if char in "0123456789":
            class_id = int(char)
            boxes.append(f"{class_id} {cx:.6f} {cy:.6f} {w / image_size[0]:.6f} {h / image_size[1]:.6f}")

        x += w + spacing

    return boxes

def add_noise_and_blur(img):
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img

# ====== 主循环：生成满足结尾条件的数据 ======
valid_suffixes = ("08", "80", "88", "008", "888","808","080")
generated = 0
index = start_index

while generated < target_count:
    # 生成 6 位数字
    digits = ''.join(random.choices("0123456789", k=6))

    # 判断是否满足结尾条件
    if not any(digits.endswith(suffix) for suffix in valid_suffixes):
        continue

    # 加载背景图
    bg_path = random.choice(background_paths)
    background = Image.open(bg_path).convert("RGB")
    image_size = background.size

    # 创建画布副本
    img = background.copy()
    img = add_noise_and_blur(img)
    draw = ImageDraw.Draw(img)

    # 左上角数字框
    left_box = (20, 20, 500, digit_font_size + 20)
    left_boxes = draw_text_with_boxes_centered(draw, digits, digit_font, left_box, image_size)

    # 右上角时间戳
    time_box_width = 700
    time_box = (image_size[0] - time_box_width - 150, 20, time_box_width, time_font_size + 20)
    now = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 1000000))
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    time_boxes = draw_text_with_boxes_centered(draw, time_str, time_font, time_box, image_size)

    # 保存图片和标注
    img_path = os.path.join(output_dir, "images", f"{index:05}.jpg")
    label_path = os.path.join(output_dir, "labels", f"{index:05}.txt")
    img.save(img_path)
    with open(label_path, 'w') as f:
        f.write('\n'.join(left_boxes + time_boxes))

    # 控制台日志
    if generated % 100 == 0:
        print(f"已生成 {generated} / {target_count} 张（以08,80,88,008,888结尾）")

    generated += 1
    index += 1

print(f"✅ 特殊样本生成完成：共 {target_count} 张，从编号 {start_index} 开始")
