from ultralytics import YOLO

# 加载预训练模型（推荐使用yolov8s.pt作为基础）
model = YOLO("yolov8m.pt")
# model = YOLO("yolov8s.pt")
# 开始训练
results = model.train(
    data="number-data.yaml",       # 数据配置文件路径
    epochs=30,             # 训练轮次
    batch=24,               # 批次大小（根据GPU显存调整）
    imgsz=640,              # 输入图像尺寸
    device=0,               # 使用GPU（0表示第一块GPU）
    name="yolov8m_number",   # 实验名称（用于保存结果）
    optimizer="Adam",       # 优化器（可选SGD、Adam等）
    lr0=0.001,              # 初始学习率
    pretrained=True,        # 是否使用预训练权重
    mosaic=1.0             # mosaic增强比例
    # mixup=0.2                        # mixup增强，适合数字泛化
)