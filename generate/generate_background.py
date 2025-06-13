from PIL import Image, ImageDraw, ImageFilter
import random
import os

# 输出背景图目录
background_output_dir = "./image/backgrounds"
os.makedirs(background_output_dir, exist_ok=True)

# 随机背景尺寸范围（宽, 高）
size_range = [(1280, 720), (1580, 742), (1920, 1080)]

# 背景图数量
background_count = 10


# 随机颜色生成函数
def random_color():
    return tuple(random.randint(100, 255) for _ in range(3))


# 背景生成函数
def generate_background(size):
    img = Image.new("RGB", size, color=random_color())
    draw = ImageDraw.Draw(img)

    # 加入几何干扰线条
    for _ in range(20):
        x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
        x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
        draw.line((x1, y1, x2, y2), fill=random_color(), width=random.randint(1, 3))

    # 加入模糊扰动
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return img


# 批量生成背景图
for i in range(background_count):
    size = random.choice(size_range)
    bg = generate_background(size)
    bg.save(os.path.join(background_output_dir, f"bg_{i + 1:02}.png"))
