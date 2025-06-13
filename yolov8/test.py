import math
import shutil
import time

import PIL
from PIL import ImageEnhance,Image
from ultralytics import YOLO
import cv2
import os
import numpy as np
import glob
from pathlib import Path



# 将图像旋转到指定角度
def rotate_image(img, degrees):
    """
    将图像旋转到指定角度
    """
    rows, cols = img.shape[:2]
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, degrees, 1)
    top_right = np.array((cols - 1, 0)) - np.array(center)
    bottom_right = np.array((cols - 1, rows - 1)) - np.array(center)
    top_right_after_rot = M[0:2, 0:2].dot(top_right)
    bottom_right_after_rot = M[0:2, 0:2].dot(bottom_right)
    new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
    new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
    offset_x = (new_width - cols) / 2
    offset_y = (new_height - rows) / 2
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    dst = cv2.warpAffine(img, M, (new_width, new_height))
    return dst


def crop_show(img,pointA,image_path):
    global rect, box, weight, height, size
    # crop_position = crop_img_position(piontA,piontB)
    senert = np.array([pointA[0], pointA[1], pointA[2], pointA[3]], dtype=np.int32)
    rect = cv2.minAreaRect(senert)
    # 获取矩形四个顶点并排序box（左上，右上，右下，左下）
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # rect
    weight, height = rect[1]
    weight = weight * 1.0
    height = height * 1.0
    src_pts = box.astype("float32")
    print(f"src_pts坐标{src_pts}")
    # src_pts = order_points_clockwise(src_pts)  # 强制按 左上、右上、右下、左下 排序
    # print(f"强制变换后的src_pts坐标{src_pts}")
    # 根据情况进行透视变换，将图片摆正
    # src_pts为原图对应坐标
    # dst_pts为转换后图像内角点对应坐标
    dst_pts = np.array([[0, height], [0, 0], [weight, 0], [weight, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (int(weight), int(height)))
    size = warped.shape
    if size[0] > size[1]:
        warped = rotate_image(warped, -90)
        temp = height
        height = weight
        weight = temp
    # 对图片进行锐化处理
    contrast = ImageEnhance.Contrast(Image.fromarray(np.uint8(warped)))
    img = contrast.enhance(1.5)
    hr_img = img.resize((int(weight), int(height)), resample=PIL.Image.BICUBIC)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_crop.jpg")
    hr_img.save(output_path)
    print(f"裁剪图像已保存至: {output_path}")

    # hr_img.show()


# 加载训练好的OBB模型
model = YOLO("/home/bz/workspace/ultralytics/runs/obb/yolov8_obb_640_20250527/weights/best.pt",task="obb")
#GPU推理
model.to('cuda')
print(model.device)

no_detect_imgs = []

input_dir = "/home/bz/桌面/虎门分拣搁架-20250526-20250528/2025_05_28"  # 替换为输入目录
output_dir = "/home/bz/桌面/虎门分拣搁架-20250526-20250528/2025-05-28_output_640"  # 替换为输出目录
no_detect_dir = os.path.join("/home/bz/桌面/虎门分拣搁架-20250526-20250528/", "2025_05_28_640_no_detections")

# input_dir = "/home/bz/桌面/EMS"  # 替换为输入目录
# output_dir = "/home/bz/桌面/EMS_OUTPUT"  # 替换为输出目录
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
os.makedirs(no_detect_dir,exist_ok=True)



# 支持的图片格式
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
# 用于存储每次推理耗时
inference_times = []


# 执行 10 次推理
for idx, image_path in enumerate(image_paths):
    start = time.time()

    # 执行推理
    results = model.predict(
        source=image_path,
        conf=0.8,  # 置信度阈值（根据需求调整）
        iou=0.75,  # NMS IoU阈值
        imgsz=640,  # 图像尺寸需与训练一致
        device='cuda:0',  # 强制使用 GPU
        save=False,  # 不保存可视化图像
        save_txt=False,  # 不保存 txt 文件
        save_conf=True,  # 不保存置信度
        project='your_output_dir',  # 输出目录
        name=f"prediction_{idx}",  # 每次的预测文件名
        exist_ok=True  # 允许覆盖已有结果
    )

    predict_end = (time.time() - start) * 1000
    print(f"推理消耗的时间{predict_end}ms")

    # 类别名列表（适配你的模型）
    names = model.names

    # 只取一张图的结果（可循环多张）
    r = results[0]
    if r.obb.cls is None or len(r.obb.cls) == 0:
        print(f"❌ 无法检测出目标")
        # 只取文件名部分
        filename = Path(image_path).name
        dst_path = os.path.join(no_detect_dir, filename)
        shutil.copy(image_path, dst_path)
        no_detect_imgs.append(image_path)
        continue


    """
    [x1, y1]  → 左上  
    [x2, y2]  → 右上  
    [x3, y3]  → 右下  
    [x4, y4]  → 左下
    
    [[738.5714721679688, 1222.819580078125], 
    [1446.947021484375, 1191.426513671875], 
    [1395.2783203125, 25.53814697265625], 
    [686.9027709960938, 56.93121337890625]]
    """

    img = cv2.imread(image_path)
    piontA = []
    # 遍历所有检测结果
    for i, (pts, cls_idx, conf) in enumerate(zip(r.obb.xyxyxyxy, r.obb.cls, r.obb.conf)):
        # 四点坐标
        points = pts.cpu().numpy().tolist()  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # 类别和置信度
        class_name = names[int(cls_idx)]
        # angle = r.obb.xywhr[4]
        if class_name == "waybill-reciver":
            piontA = points
        else:
            print("没有检测到目标，将跳过")
            continue
        confidence = float(conf)
        angle = r.obb.xywhr[0][4].cpu().item()  # 旋转角度（弧度）
        print(f"目标 {i}: 类别 = {class_name}, 置信度 = {confidence:.2f}, 四点 = {points},弧度={angle}")

    # 裁剪展示
    crop_start = time.time()
    crop_show(img,piontA,image_path)
    crop_end = (time.time() - crop_start) * 1000
    print(f"crop裁剪消耗时间{crop_end}ms")

    end = (time.time() - start) * 1000  # 计算每次推理时间（ms）
    inference_times.append(end)
    print(f"第{idx + 1}次推理总耗时: {end:.2f}ms")
print(f"无法检测的目标{no_detect_imgs}")
