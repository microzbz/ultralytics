import math
import time

import PIL
from PIL import ImageEnhance,Image
from pydantic import UUID1

from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


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



# 加载训练好的OBB模型
model = YOLO("/home/bz/workspace/ultralytics/yolov8/runs/obb/yolov8_obb3/weights/best.pt",task="obb")
#GPU推理
model.to('cuda')
print(model.device)

# 定义输入输出路径
# input_image = "/home/bz/workspace/ultralytics/yolov8/datasets/split-dataset/test/images/22e7a7dc-52.jpg"  # 输入图像路径
# input_image = "/home/bz/workspace/ultralytics/yolov8/datasets/split-dataset/train/images/d06353a5-54.jpg"
# input_image = "/home/bz/workspace/ultralytics/yolov8/datasets/split-dataset/train/images/bfad71d4-38.jpg"


#法院专递
input_image = "/home/bz/workspace/ultralytics/yolov8/datasets/split-dataset/train/images/bfdfa51f-23.jpg"

output_dir = "obb_results"  # 结果保存目录
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

# 用于存储每次推理耗时
inference_times = []


def get_center_of_box(points):
    """ 获取四个点的中心点 """
    return np.mean(np.array(points, dtype=np.float32), axis=0).reshape(1, 1, 2)


def get_relative_position(A_center_transformed, B_center_transformed):
    """
    判断 B 相对 A 的位置: 左、右、上、下
    :param A_center_transformed: A 中心点的变换后的坐标 (x_A, y_A)
    :param B_center_transformed: B 中心点的变换后的坐标 (x_B, y_B)
    :return: 相对位置字符串
    """
    x_A, y_A = A_center_transformed
    x_B, y_B = B_center_transformed

    # 判断左、右、上下
    position = ''

    # 判断左右位置
    if x_B < x_A:
        position += 'left '
    elif x_B > x_A:
        position += 'right '
    else:
        position += 'center '

    # # 判断上下位置
    # if y_B < y_A:
    #     position += 'top'
    # elif y_B > y_A:
    #     position += 'bottom'
    # else:
    #     position += 'center'

    return position.strip()


import cv2





def get_direction(A, B):
    """
    判断点B相对于点A的方向（基于图像坐标系，y轴向下）
    :param A: 元组格式的中心点坐标，如 (x, y)
    :param B: 元组格式的中心点坐标，如 (x, y)
    :return: 方向字符串，四种可能：左上、左下、右上、右下
    """
    Ax, Ay = A[0][0]
    Bx, By = B[0][0]

    dx = Bx - Ax
    dy = By - Ay

    # 判断水平方向：dx>0右，dx<0左
    horizontal = "右" if dx > 0 else "左"
    # 判断垂直方向：dy<0上，dy>0下（图像坐标系y轴向下）
    vertical = "上" if dy < 0 else "下"

    return f"{horizontal}{vertical}"

# x1,y1 到 x2,y2的方向按比例截取
def get_point_on_line(x1, y1, x2, y2, t):
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return [x, y]

def find_index_in_polygon(point, polygon):
    print(len(polygon))
    for idx in range(len(polygon)):
        if (polygon[idx]==point).all(): # 如果当前顶点和传入的点相同
            return idx
    return  -1

#获取最近的点和坐标
def find_closest_point_with_index(target, points):
    """
    找到目标点在点列表中的最近点，并返回其索引、坐标和距离。

    参数:
        target (list): 目标点 [x, y]
        points (list): 点列表 [[x1, y1], [x2, y2], ...]

    返回:
        closest_index (int): 最近点在列表中的索引
        closest_point (list): 最近点的坐标
        min_distance (float): 最近距离
    """
    min_distance = float('inf')
    closest_index = -1
    closest_point = None

    for index, point in enumerate(points):
        dx = point[0] - target[0]
        dy = point[1] - target[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance < min_distance:
            min_distance = distance
            closest_point = point
            closest_index = index

    return closest_index, closest_point

def crop_img_postion_fyzd(pointA,pointB):
    left_top = piontA[0]
    right_top = pointA[1]
    right_bottom = pointA[2]
    left_bottom = pointA[3]

    a_center = get_center_of_box(pointA)
    b_center = get_center_of_box(pointB)

    index,closest_point = find_closest_point_with_index(b_center[0][0],pointA)
    print(f"距离最近的点是，index={index} point={closest_point}")

    postion = get_direction(a_center, b_center)
    print(f"方向是{postion}")

    if index ==0:
        right_top = pointA[0]
        left_top = piontA[1]
        left_bottom = piontA[2]
        right_bottom = piontA[3]
    elif index ==1:
        right_top = pointA[1]
        left_top = piontA[2]
        left_bottom = piontA[3]
        right_bottom = piontA[0]
    elif index ==2:
        right_top = pointA[2]
        left_top = piontA[3]
        left_bottom = piontA[0]
        right_bottom = piontA[1]
    elif index ==3:
        right_top = pointA[3]
        left_top = piontA[0]
        left_bottom = piontA[1]
        right_bottom = piontA[2]

    return crop_position_by_fyzd(left_bottom, left_top, right_bottom, right_top)


def crop_position_by_fyzd(left_bottom, left_top, right_bottom, right_top):
    half_left_top_right_top = get_point_on_line(left_top[0], left_top[1], right_top[0], right_top[1], 0.43)
    half_right_bottom_right_bottom = get_point_on_line(left_bottom[0], left_bottom[1], right_bottom[0],
                                                       right_bottom[1], 0.43)
    new_right_top = get_point_on_line(right_top[0], right_top[1],
                                      right_bottom[0], right_bottom[1], 0.2)
    new_right_bottom = get_point_on_line(right_top[0], right_top[1],
                                         right_bottom[0], right_bottom[1], 0.7)

    new_left_top = get_point_on_line(half_left_top_right_top[0], half_left_top_right_top[1], half_right_bottom_right_bottom[0], half_right_bottom_right_bottom[1], 0.2)
    new_left_bottom = get_point_on_line(half_left_top_right_top[0], half_left_top_right_top[1], half_right_bottom_right_bottom[0], half_right_bottom_right_bottom[1], 0.7)
    return [new_left_top, new_right_top, new_right_bottom, new_left_bottom]







#国内特快专递裁剪
def crop_img_position(pointA,pointB):
    left_top = piontA[0]
    right_top = pointA[1]
    right_bottom = pointA[2]
    left_bottom = pointA[3]

    a_center = get_center_of_box(pointA)
    b_center = get_center_of_box(pointB)

    index,closest_point = find_closest_point_with_index(b_center[0][0],pointA)
    print(f"距离最近的点是，index={index} point={closest_point}")
    if index ==0:
        right_top = pointA[0]
        left_top = piontA[1]
        left_bottom = piontA[2]
        right_bottom = piontA[3]
    elif index ==1:
        right_top = pointA[1]
        left_top = piontA[2]
        left_bottom = piontA[3]
        right_bottom = piontA[0]
    elif index ==2:
        right_top = pointA[2]
        left_top = piontA[3]
        left_bottom = piontA[0]
        right_bottom = piontA[1]
    elif index ==3:
        right_top = pointA[3]
        left_top = piontA[0]
        left_bottom = piontA[1]
        right_bottom = piontA[2]

    postion = get_direction(a_center,b_center)
    print(f"方向是{postion}")
    if "右" in postion and "上" in postion:
        # 说明是正向,按照
        return crop_position(left_bottom, left_top, right_bottom, right_top)
    if "左" in postion and "下" in postion:
        return crop_position(left_bottom, left_top, right_bottom, right_top)
    return pointA


def crop_position(left_bottom, left_top, right_bottom, right_top):
    new_left_top = get_point_on_line(left_top[0], left_top[1], left_bottom[0], left_bottom[1], 0.3)
    new_left_bottom = get_point_on_line(left_top[0], left_top[1], left_bottom[0], left_bottom[1], 0.7)
    half_left_top_right_top = get_point_on_line(left_top[0], left_top[1], right_top[0], right_top[1], 0.57)
    half_right_bottom_right_bottom = get_point_on_line(left_bottom[0], left_bottom[1], right_bottom[0],
                                                       right_bottom[1], 0.57)
    new_right_top = get_point_on_line(half_left_top_right_top[0], half_left_top_right_top[1],
                                      half_right_bottom_right_bottom[0], half_right_bottom_right_bottom[1], 0.3)
    new_right_bottom = get_point_on_line(half_left_top_right_top[0], half_left_top_right_top[1],
                                         half_right_bottom_right_bottom[0], half_right_bottom_right_bottom[1], 0.7)
    return [new_left_top, new_right_top, new_right_bottom, new_left_bottom]


def crop_show(img,pointA,pointB):
    global rect, box, weight, height, size
    # crop_position = crop_img_position(piontA,piontB)
    crop_position = crop_img_postion_fyzd(piontA,piontB)
    print(f"crop_img_position ={crop_position}")
    senert = np.array([crop_position[0], crop_position[1], crop_position[2], crop_position[3]], dtype=np.int32)
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
    hr_img.save(f"hr_img{time.time_ns()}.jpg")
    # hr_img.show()

    height, width = hr_img.size

    print("宽度:", width)
    print("高度:", height)

    a_center = get_center_of_box(pointA)
    b_center = get_center_of_box(pointB)

    print("a_center:", a_center)
    print("b_center:", b_center)

    # 对 A 和 B 的中心点都应用透视变换
    A_center_transformed = cv2.perspectiveTransform(a_center, M)[0][0]
    B_center_transformed = cv2.perspectiveTransform(b_center, M)[0][0]

    print("A_center_transformed:", A_center_transformed)
    print("B_center_transformed:", B_center_transformed)

    position = get_relative_position(A_center_transformed, B_center_transformed)
    print(f"position: {position}")
    # if position == "right":
    #     cropped_image = crop_image_by_position(hr_img, 'top', top_ratio=0.35, bottom_ratio=0.71, left_ratio=0.0,
    #                                            right_ratio=0.57)
    #     # 显示裁剪后的图像
    #     cv2.imshow('Cropped Image', cropped_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()



def label_show():
    global pts, img
    # 转为四个点的数组
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    img = cv2.imread(input_image)

    # 画多边形框
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 添加标签（类别 + 置信度）
    label = "waybill 0.95"
    cv2.putText(img, label, (pts[0][0][0], pts[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 转换为 RGB（matplotlib 用）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示交互窗口
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title("OBB Bounding Box")
    plt.axis('on')
    plt.tight_layout()
    plt.show()


# 执行 10 次推理
for j in range(1):
    start = time.time()

    # 执行推理
    results = model.predict(
        source=input_image,
        conf=0.25,  # 置信度阈值（根据需求调整）
        iou=0.45,  # NMS IoU阈值
        imgsz=640,  # 图像尺寸需与训练一致
        device='cuda:0',  # 强制使用 GPU
        save=False,  # 不保存可视化图像
        save_txt=False,  # 不保存 txt 文件
        save_conf=False,  # 不保存置信度
        project='your_output_dir',  # 输出目录
        name=f"prediction_{j}",  # 每次的预测文件名
        exist_ok=True  # 允许覆盖已有结果
    )

    predict_end = (time.time() - start) * 1000
    print(f"推理消耗的时间{predict_end}ms")

    # 类别名列表（适配你的模型）
    names = model.names

    # 只取一张图的结果（可循环多张）
    r = results[0]


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

    img = cv2.imread(input_image)
    piontA = []
    piontB = []
    # 遍历所有检测结果
    for i, (pts, cls_idx, conf) in enumerate(zip(r.obb.xyxyxyxy, r.obb.cls, r.obb.conf)):
        # 四点坐标
        points = pts.cpu().numpy().tolist()  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # 类别和置信度
        class_name = names[int(cls_idx)]
        # angle = r.obb.xywhr[4]
        if class_name == "waybill":
            piontA = points
        elif class_name == "GNTK":
            piontB = points
        elif class_name == "FYZD":
            piontB = points

        confidence = float(conf)

        angle = r.obb.xywhr[0][4].cpu().item()  # 旋转角度（弧度）

        print(f"目标 {i}: 类别 = {class_name}, 置信度 = {confidence:.2f}, 四点 = {points},弧度={angle}")
        #标注框展示
        # label_show()

    # 裁剪展示
    crop_start = time.time()
    crop_show(img,piontA,piontB)
    crop_end = (time.time() - crop_start) * 1000
    print(f"crop裁剪消耗时间{crop_end}ms")

    end = (time.time() - start) * 1000  # 计算每次推理时间（ms）
    inference_times.append(end)
    print(f"第{j + 1}次推理总耗时: {end:.2f}ms")

# 计算平均时间
average_time = sum(inference_times) / len(inference_times)
print(f"10次推理的平均耗时: {average_time:.2f}ms")