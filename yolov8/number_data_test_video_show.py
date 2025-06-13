from ultralytics import YOLO
import cv2
#用于显示检测效果
# 加载模型
model = YOLO("/home/bz/workspace/ultralytics/runs/detect/yolov8_number9/weights/best.pt")

# 推理视频或图片（使用 stream=True 逐帧处理视频）
results = model.predict(
    source="/home/bz/追溯/Channel_44_T_20250531_083113_T_20250531_083613.mp4",
    stream=True,
    conf=0.9,
    show=False)

# 获取类别名
names = model.names

for result in results:
    frame = result.orig_img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = names[cls_id]

        # 绘制框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 标签绘制在目标框下方偏移 40 像素
        offset_y = 60
        label_position = (x1, min(frame.shape[0] - 10, y2 + offset_y))
        cv2.putText(frame, label, label_position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5, color=(0, 255, 255), thickness=3)

    # 显示每一帧
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
