import cv2
import time
from ultralytics import YOLO
model = YOLO("/home/bz/workspace/ultralytics/runs/detect/yolov8m_number2/weights/best.pt")
start = time.time()
result = model.predict(
    source="/home/bz/图片/",
    save=True,
    save_txt=False,
    project="number_results",
    name="number_video_test_output_image",
    show_conf=False,
    show_labels=True,
    conf=0.5
)
total_cost  = (time.time() - start)*1000
print(f"检测完成，总消耗时间{total_cost}ms")
