from ultralytics import YOLO

# 加载预训练模型（推荐使用yolov8s.pt作为基础）
# model = YOLO("yolov8s.pt")

model = YOLO("yolov8l-obb.pt",task="obb")
# 开始训练
results = model.train(
    data="data.yaml",       # 数据配置文件路径
    epochs=100,             # 训练轮次
    batch=16,               # 批次大小（根据GPU显存调整）
    imgsz=640,              # 输入图像尺寸
    device=0,               # 使用GPU（0表示第一块GPU）
    name="yolov8_obb_640_20250527",   # 实验名称（用于保存结果）
    optimizer="Adam",       # 优化器（可选SGD、Adam等）
    lr0=0.0001,              # 初始学习率
    pretrained=True         # 是否使用预训练权重
)