from flask import Flask, request, jsonify
import cv2
import time
from ultralytics import YOLO
from yolov8.video_increment_check import VideoIncrementChecker
import os

app = Flask(__name__)

# 预加载模型（启动时加载，避免每次请求加载）
# model = YOLO("/home/bz/workspace/ultralytics/runs/detect/yolov8_number21/weights/best.pt")

model = YOLO("/home/bz/workspace/ultralytics/runs/detect/yolov8m_number2/weights/best.pt")


# 快速读取大文件最后一行
def read_last_line(filepath, chunk_size=1024):
    with open(filepath, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        offset = min(chunk_size, file_size)
        f.seek(-offset, 2)
        chunk = f.read(offset)
        lines = chunk.split(b'\n')
        if len(lines) >= 2:
            return lines[-1].decode("utf-8") if lines[-1] else lines[-2].decode("utf-8")
        return lines[0].decode("utf-8")

def process_batch_results(results, frame_ids,total_frame,conf_threshold=0.9, increment_checker=None):
    all_frame_results = []

    for i, r in enumerate(results):
        frame_no = frame_ids[i]
        boxes = r.boxes

        if boxes is None or len(boxes.conf) == 0:
            print(f"[Frame {frame_no}] ❌ 无检测结果")
            all_frame_results.append({
                "frame": frame_no,
                "status": "abnormal",
                "detections": []
            })
            continue

        detections = []
        frame_ok = True

        for cls_id, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
            cls_id = int(cls_id.item())
            class_name = model.names.get(cls_id, f"unknown({cls_id})")
            conf_val = round(float(conf.item()), 2)
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            detections.append({
                "class": class_name,
                "conf": conf_val,
                "box": (x1, y1, x2, y2)
            })
            if conf_val < conf_threshold:
                frame_ok = False

        detections.sort(key=lambda d: d["box"][0])  # 按 x1 排序

        status = "normal" if frame_ok else "abnormal"

        if increment_checker is not None:
            if not increment_checker.update(detections, frame_no=frame_no):
                status = "abnormal"

        all_frame_results.append({
            "total_frame":total_frame,
            "frame": frame_no,
            "status": status,
            "detections": detections
        })

    return all_frame_results

def detect_video(video_path):
    BATCH_SIZE = 10
    batch_frames = []
    frame_ids = []
    frame_id = 0
    cap = cv2.VideoCapture(video_path)
    all_results = []
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    increment_checker = VideoIncrementChecker(tail_len=14)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h = frame.shape[0]  #高度 像素
        w = frame.shape[1]  # 宽度，单位：像素
        print(f"图片大小为高{h}宽{w}")
        cropped = frame[:h // 10, :]  # 上面1/10区域
        batch_frames.append(cropped)
        frame_ids.append(frame_id)
        frame_id += 1

        if len(batch_frames) == BATCH_SIZE:
            results = model.predict(batch_frames, conf=0.5, verbose=False)
            batch_results = process_batch_results(results,frame_ids, total_frame,increment_checker=increment_checker)
            all_results.extend(batch_results)
            batch_frames = []
            frame_ids = []

    if batch_frames:
        results = model.predict(batch_frames, conf=0.5, verbose=False)
        batch_results = process_batch_results(results, frame_ids,total_frame, increment_checker=increment_checker)
        all_results.extend(batch_results)

    cap.release()
    return all_results



# def is_strictly_increasing(nums):
#     return all(x < y for x, y in zip(nums, nums[1:]))
#
# def check_detections_part(detections):
#     if len(detections) < 14:
#         # 长度不够，无法取到倒数第14个
#         return False
#
#     # 取从 detections[0] 到 detections[-14]（含）这部分
#     sub_list = detections[0:len(detections)-14+1]  # python切片不含end, 所以+1
#
#     nums = []
#     for det in sub_list:
#         cls_str = det["class"]
#         if len(cls_str) <= 14:
#             return False
#         num_str = cls_str[:len(cls_str) - 14]
#         try:
#             num = int(num_str)
#             nums.append(num)
#         except ValueError:
#             return False
#     print(nums)
#     return is_strictly_increasing(nums)
#
# def process_batch_results(results, frame_ids, conf_threshold=0.9):
#     all_frame_results = []
#     for i, r in enumerate(results):
#         frame_no = frame_ids[i]
#         boxes = r.boxes
#
#         if boxes is None or len(boxes.conf) == 0:
#             all_frame_results.append({
#                 "frame": frame_no,
#                 "status": "异常 - 无检测结果",
#                 "detections": []
#             })
#             continue
#
#         detections = []
#         frame_ok = True
#
#         for cls_id, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
#             cls_id = int(cls_id.item())
#             class_name = model.names.get(cls_id, f"unknown({cls_id})")
#             conf_val = float(conf.item())
#             x1, y1, x2, y2 = map(int, xyxy.tolist())
#             detections.append({
#                 "class": class_name,
#                 "conf": conf_val,
#                 "box": (x1, y1, x2, y2)
#             })
#             if conf_val < conf_threshold:
#                 frame_ok = False
#
#         detections.sort(key=lambda d: d["box"][0])  # 按 x1 排序
#         status = "normal" if frame_ok else "abnormal"
#         if check_detections_part(detections) is False:
#             status = "abnormal"
#         all_frame_results.append({
#             "frame": frame_no,
#             "status": status,
#             "detections": detections
#         })
#     return all_frame_results
#
#
#
# def detect_video(video_path):
#     BATCH_SIZE = 10
#     batch_frames = []
#     frame_ids = []
#     frame_id = 0
#     cap = cv2.VideoCapture(video_path)
#     all_results = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         h = frame.shape[0]
#         cropped = frame[:h // 10, :]  # 上面1/10区域
#         batch_frames.append(cropped)
#         frame_ids.append(frame_id)
#         frame_id += 1
#
#         if len(batch_frames) == BATCH_SIZE:
#             results = model.predict(batch_frames, conf=0.85, verbose=False)
#             batch_results = process_batch_results(results, frame_ids)
#             all_results.extend(batch_results)
#             batch_frames = []
#             frame_ids = []
#
#     # 处理剩余帧
#     if batch_frames:
#         results = model.predict(batch_frames, conf=0.85, verbose=False)
#         batch_results = process_batch_results(results, frame_ids)
#         all_results.extend(batch_results)
#
#     cap.release()
#     return all_results


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    video_path = data.get("video_path")
    if not video_path:
        return jsonify({"error": "参数缺失: video_path"}), 400
    # video_id = data.get("video_id")
    # if not video_id:
    #     return jsonify({"error": "参数缺失: video_id"}), 400
    #
    # txt_path = video_path + video_id+".txt"
    # # 如果结果文件已存在，直接读取最后一行
    # if os.path.exists(txt_path):
    #     last_result = read_last_line(txt_path)
    #     return last_result

    start_time = time.time()
    results = detect_video(video_path)
    cost = (time.time() - start_time) * 1000
    return jsonify({
        "processing_time_ms": cost,
        "results": results
    })


if __name__ == '__main__':
    # 监听所有地址，方便远程访问
    app.run(host='0.0.0.0', port=5000, debug=False)
