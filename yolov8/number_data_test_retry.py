import gc
from datetime import datetime
from threading import Thread

import torch
from flask import Flask, request, jsonify
import cv2

from ultralytics import YOLO
from video_increment_check import VideoIncrementChecker
import json
import os
import copy
import config
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue

@dataclass
class DetectTask:
    video_name: str
    result_path: str
    video_path: str
    video_status: dict

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=20)  # 设置并发批次数，根据CPU/GPU可调

task_queue = Queue()

# 预加载模型（启动时加载，避免每次请求加载）
model = YOLO(config.model_path)
conf = config.confidence
# 全局状态字典,保存每个视频的检测状态
video_status_map = {}
task_set = set()  # 直接用 video_path 去重


def detect_worker():
    while True:
        task = task_queue.get()
        try:
            if task is None:
                # print("[WARN] 收到空任务，跳过处理")
                # 标记任务完成，继续下一轮循环
                continue
            running_async(task.result_path, task.video_name, task.video_path, task.video_status)
        except Exception as e:
            print(f"[ERROR] 检测任务失败: {e}")
        finally:
            task_set.discard(task.video_path)  # 处理完成后释放占用
            task_queue.task_done()


def get_last_two_lines(result_file):
    """读取 JSONL 文件最后两行，返回解析后的 JSON 对象"""
    last_two_lines = []

    try:
        with open(result_file, 'rb') as f:
            f.seek(0, os.SEEK_END)
            pointer = f.tell() - 1
            buffer = bytearray()

            while pointer >= 0 and len(last_two_lines) < 2:
                f.seek(pointer)
                byte = f.read(1)

                if byte == b'\n':
                    if buffer:
                        # 反转字节顺序后解码，避免字符串倒序
                        line = buffer[::-1].decode('utf-8').strip()
                        last_two_lines.insert(0, line)
                        buffer = bytearray()
                else:
                    buffer.append(byte[0])  # 这里append，后续反转

                pointer -= 1

            # 文件头还有残留数据
            if buffer:
                line = buffer[::-1].decode('utf-8').strip()
                last_two_lines.insert(0, line)

        if not last_two_lines:
            return None, None

        # 读取出来的最后两行字符串
        print("Last two lines:", last_two_lines)

        last_json = json.loads(last_two_lines[-1])
        is_status_line = "process_status" in last_json

        if is_status_line and len(last_two_lines) >= 2:
            last_frame_result = json.loads(last_two_lines[-2])
        else:
            last_frame_result = last_json
            last_json = None

        return last_frame_result, last_json

    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None

def build_response(last_frame_result, status_info):
    response = {}

    if status_info:
        response.update(status_info)
    if last_frame_result:
        response["last_frame_result"] = last_frame_result

    return response

@app.route("/progress", methods=["POST"])
def video_progress():
    data = request.json
    video_id = data.get("video_id")
    if not video_id:
        return jsonify({"error": "缺少 video_id 参数"}), 400

    result_file = os.path.join(config.save_result_dir, f"{video_id}.json")
    if not os.path.isfile(result_file):
        return jsonify({"error": "结果文件不存在"}), 404

    last_frame_result, status_info = get_last_two_lines(result_file)

    if not last_frame_result:
        return jsonify({"error": "读取失败"}), 500
    local_video_status = copy.deepcopy(video_status_map.get(video_id, {}))

    if video_id in local_video_status:
        local_video_status[video_id]["processed_frames"] = last_frame_result["frame"]
        local_video_status[video_id].update({
            "last_frame_result": last_frame_result
        })
        return jsonify(local_video_status[video_id])
    else:
        response_data = build_response(last_frame_result, status_info)
        return jsonify(response_data)


def run_and_check(batch_frames, frame_ids, increment_checker):
    attempt = 1
    # 第一次：批量检测
    results = model.predict(batch_frames, conf=conf, verbose=False)
    batch_results = process_batch_results(results, frame_ids, increment_checker=increment_checker, frames=batch_frames)
    #清除缓存和张量引用
    del results
    torch.cuda.empty_cache()
    if all(res["status"] == "normal" for res in batch_results):
        return batch_results, True
    #有问题的帧多次检测
    status = True
    while attempt < config.max_retry:
        if increment_checker.error_buffer is None:
            return  batch_results, False
        print(f"{frame_ids}正在重试{attempt}次")
        error_frame_list = increment_checker.error_buffer
        print("需要重试的帧号列表:", [f[1] for f in error_frame_list])
        #将维护的last_number设置为 错误帧的上上帧的上一个数字
        increment_checker.last_number = error_frame_list[-increment_checker.retry_frame_size][3]
        status = True
        for error_frame in error_frame_list:
            frame_no = error_frame[1]
            frame = error_frame[0]
            frame_number = error_frame[2]
            print(f"当前帧正在检测{frame_no}")
            promote_conf = conf
            #长度超过7,说明置信度太低,调高,低于7,说明置信度设置太高,得降低
            if len(str(frame_number)) > 7:
                promote_conf = conf + 0.1
            if len(str(frame_number))<7:
                promote_conf = conf - 0.5
            results = model.predict([frame], conf=promote_conf, verbose=False)
            re_result = process_batch_results(results,[frame_no],increment_checker=increment_checker,frames=[frame],error_retry=True)
            # 清除缓存和张量引用
            del results
            torch.cuda.empty_cache()
            if isinstance(re_result, list):
                re_result = re_result[0]
            print(f"检测结果{re_result}")
            if re_result["status"] == "normal":
                # 替换原来的 batch_results 中对应 frame_no 的结果
                for i, orig_res in enumerate(batch_results):
                    if orig_res.get("frame") == frame_no:
                        batch_results[i] = re_result
                        break
            else:
                status = False
        if status is True:
            print(f"{frame_ids}重试成功,lastnumber{increment_checker.last_number}")
            break
        attempt += 1
    return batch_results,status


def process_batch_results(results, frame_ids, conf_threshold=0.9, increment_checker=None,frames=None,error_retry=False):
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
            conf_val = round(float(conf.item()), 3)
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            # frame_height = frames[i].shape[0] if frames is not None else None
            # # 只保留上半部分的检测框
            # if frame_height is not None and y2 > frame_height // 2:
            #     continue
            detections.append({
                "class": class_name,
                "conf": conf_val,
                "box": (x1, y1, x2, y2)
            })
            # if conf_val < conf_threshold:
            #     frame_ok = False

        detections.sort(key=lambda d: d["box"][0])  # 按 x1 排序

        status = "normal" if frame_ok else "abnormal"

        if increment_checker is not None:
            if not increment_checker.update(detections, frame_no=frame_no,frame=frames[i],error_retry=error_retry):
                status = "abnormal"

        number_res,time_res= get_numbers(detections)
        all_frame_results.append({
            "frame": frame_no,
            "status": status,
            "result":number_res,
            "time_result":time_res
        })
        del boxes

    return all_frame_results

def get_numbers(detections):
    main_part = detections[:-14]
    new_number_str = ''.join(d['class'] for d in main_part)
    new_number = int(new_number_str)

    timestamp_part = detections[-14:]
    timestamp_str = ''.join(d['class'] for d in timestamp_part)
    # 转换为 datetime 对象
    dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    new_dt = dt.strftime("%Y-%m-%d %H:%M:%S")

    return new_number,new_dt

def detect_video(video_path,save_result_dir,video_name):
    result_path = os.path.join(save_result_dir, f"{video_name}.json")
    # 如果文件已存在，直接从文件或者本地map返回结果
    if os.path.exists(result_path):
        last_frame_result, status_info = get_last_two_lines(result_path)
        if not last_frame_result and video_name in video_status_map:
            return video_status_map
        local_video_status = copy.deepcopy(video_status_map)

        if video_name in local_video_status:
            local_video_status[video_name]["processed_frames"] = last_frame_result["frame"]
            local_video_status[video_name].update({
                "last_frame_result": last_frame_result
            })
            return jsonify(local_video_status[video_name])
        else:
            response_data = build_response(last_frame_result, status_info)
            return jsonify(response_data)

    # 文件不存在，需要初始化
    video_status_map[video_name] = {
        "detect_status": "normal",
        "process_status":"processing",
        "processed_frames": 0,
        "total_frames": int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    }
    if video_path in task_set:
        print(f"已存在当前{video_path}任务")
        return video_status_map[video_name]

    task_set.add(video_path)
    task = DetectTask(
        video_name=video_name,
        result_path=result_path,
        video_path=video_path,
        video_status=video_status_map[video_name]
    )
    task_queue.put(task)
    # 异步执行
    # executor.submit(running_async, result_path, video_name, video_path,video_status_map[video_name])
    return video_status_map[video_name]


def running_async(result_path, video_name, video_path,status):
    try:
        print(f"开始执行{video_name}检测任务....")
        # 再打开文件写入
        f_out = open(result_path, "w", encoding="utf-8")
        batch_frames = []
        frame_ids = []
        frame_id = 0
        cap = cv2.VideoCapture(video_path)
        all_results = []
        increment_checker = VideoIncrementChecker(tail_len=14, history_len=config.batch_size * 2,
                                                  retry_frame_size=config.batch_size)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            status["process_status"] = "processing"
            h = frame.shape[0]
            cropped = frame[:h // 10, :]
            batch_frames.append(cropped)
            frame_ids.append(frame_id)
            frame_id += 1
            status["processed_frames"] += 1

            if len(batch_frames) == config.batch_size:
                #加入队列
                batch_results, success = run_and_check(batch_frames, frame_ids, increment_checker)
                if success is False:
                    status["detect_status"] = "abnormal"
                all_results.extend(batch_results)
                # ✅ 将结果逐帧写入文件，一行一个 JSON
                for result in batch_results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                batch_frames = []
                frame_ids = []
        # 处理剩余帧
        if batch_frames:
            batch_results, success = run_and_check(batch_frames, frame_ids, increment_checker)
            if success is False:
                status["detect_status"] = "abnormal"
            all_results.extend(batch_results)
            for result in batch_results:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        # 视频状态结果保存
        status["process_status"] = "finished"
        f_out.write(json.dumps(status, ensure_ascii=False) + "\n")
        cap.release()
        f_out.close()
        if video_name in video_status_map:
            del video_status_map[video_name]
    finally:
        gc.collect()

@app.route("/detect", methods=["POST"])
def detect_api():
    data = request.json
    video_path = data.get("video_path")
    video_name = data.get("video_id")
    if not video_path:
        return jsonify({"error": "缺失参数: video_path"}), 400
    if not video_name:
        return jsonify({"error": "缺失参数: video_name"}), 400
    result_dir = config.save_result_dir
    os.makedirs(result_dir, exist_ok=True)  # 如果不存在就创建
    full_video_path = os.path.join(video_path, video_name)
    print(f"开始处理视频{full_video_path}")
    results = detect_video(full_video_path,result_dir,video_name)
    print(f"{full_video_path}处理完成")
    return results

if __name__ == "__main__":
    Thread(target=detect_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=config.port, debug=False)
