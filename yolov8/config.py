# config.py 全局配置信息
port = 5000
debug = False
#模型设置
#模型文件配置路径
model_path = "/home/bz/workspace/ultralytics/runs/detect/yolov8_number22/weights/best.pt"
confidence = 0.6
#最大重试次数
max_retry = 10
#批量检测大小
batch_size = 8

#存储结果设置,容器部署,挂载一个目录共享到容器内部,设置为容器内部目录
save_result_dir = "/home/bz/result"
