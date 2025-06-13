import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

# ========== 配置 ==========
output_image_dir = "./digit_cards/images"
output_label_dir = "./digit_cards/labels"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

font_path = "/home/bz/workspace/ultralytics/generate/Perfect DOS VGA 437.ttf"
font_size = 90
font = ImageFont.truetype(font_path, font_size)

image_width, image_height = 200, 200
samples_per_digit = 1000  # 每个数字生成多少张

background_colors = [
    (255, 255, 255), (240, 240, 240), (255, 250, 240),
    (230, 245, 255), (250, 235, 255), (245, 255, 245),
]
text_color = (255, 0, 0)  # 红色

# ========== 增强函数 ==========
def apply_blur(img):
    if random.random() < 0.7:
        radius = random.uniform(0.5, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img

def apply_salt_pepper(img, amount=0.004):
    np_img = np.array(img)
    num_noise = int(amount * np_img.size)
    for _ in range(num_noise):
        x = random.randint(0, np_img.shape[0] - 1)
        y = random.randint(0, np_img.shape[1] - 1)
        if random.random() < 0.5:
            np_img[x, y] = 255
        else:
            np_img[x, y] = 0
    return Image.fromarray(np_img)

def apply_stripes(img, stripe_count=5):
    draw = ImageDraw.Draw(img)
    for _ in range(stripe_count):
        y = random.randint(0, image_height - 1)
        stripe_height = random.randint(1, 3)
        color = (random.randint(100, 200),) * 3  # 灰条纹
        draw.rectangle([0, y, image_width, y + stripe_height], fill=color)
    return img

def apply_brightness(img):
    if random.random() < 0.7:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.5, 1.5)
        img = enhancer.enhance(factor)
    return img

def apply_all_augmentations(img):
    img = apply_blur(img)
    img = apply_brightness(img)
    # img = apply_stripes(img)
    img = apply_salt_pepper(img)
    return img

# ========== 主循环 ==========
for digit_char in "0123456789":
    for i in range(samples_per_digit):
        bg_color = random.choice(background_colors)
        img = Image.new("RGB", (image_width, image_height), bg_color)
        draw = ImageDraw.Draw(img)

        # 文字居中
        bbox = draw.textbbox((0, 0), digit_char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (image_width - text_width) // 2
        y = (image_height - text_height) // 2
        draw.text((x, y), digit_char, font=font, fill=text_color)

        # 加强图像增强
        img = apply_all_augmentations(img)

        # YOLO 标签
        cx = (x + text_width / 2) / image_width
        cy = (y + text_height / 2) / image_height
        w = text_width / image_width
        h = text_height / image_height
        label = f"{int(digit_char)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

        filename = f"{digit_char}_{i:04d}"
        img.save(os.path.join(output_image_dir, f"{filename}.png"))
        with open(os.path.join(output_label_dir, f"{filename}.txt"), "w") as f:
            f.write(label + "\n")
