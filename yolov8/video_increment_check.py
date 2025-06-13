
from collections import deque

class VideoIncrementChecker:
    def __init__(self, tail_len=14, history_len=12,retry_frame_size=6):
        self.tail_len = tail_len
        self.last_number = None
        self.history = deque(maxlen=history_len)  # 存储最近 N 帧 (frame_id, number)
        self.error_buffer = []  # 存储异常帧（用于重检）
        self.retry_frame_size = retry_frame_size
        if retry_frame_size <3:
            self.retry_frame_size = 3
        # self.retry_frame_size = history_len

    def truncate_and_get_number(self,detections, tail_len=14, max_digit_len=7):
        # 拆分 detections
        main_part = detections[:-tail_len] if tail_len < len(detections) else []
        tail_part = detections[-tail_len:] if tail_len < len(detections) else detections

        while len(''.join(d['class'] for d in main_part)) > max_digit_len and len(main_part) > 0:
            # 剔除非 tail 中置信度最低项
            min_index = min(range(len(main_part)), key=lambda i: main_part[i].get("conf", 1.0))
            removed = main_part.pop(min_index)
            print(f"剔除非 tail 区域置信度最低项: {removed}")

        new_detections = main_part + tail_part
        new_number_str = ''.join(d['class'] for d in main_part)
        try:
            new_number = int(new_number_str)
        except ValueError:
            print(f"处理后仍无法转为数字: {new_number_str}")
            return new_detections, None

        return new_detections, new_number

    def update(self, detections, frame_no=None,frame=None,error_retry=False):
        parts = []
        if len(detections) <= self.tail_len:
            print(f"[Frame {frame_no}] class 长度不足，无法提取有效部分: {detections}")
            return False
        for det in detections[:-self.tail_len]:
            class_str = det.get("class", "")
            parts.append(class_str)

        try:
            current_number = int("".join(parts))
        except ValueError:
            print(f"[Frame {frame_no}] 拼接后无法转换为数字: {''.join(parts)}")
            return False
        #错误重试检测时,不更新历史数据
        if error_retry is False:
            self.history.append((frame, frame_no, current_number, self.last_number))

        if self.last_number is None:
            print(f"[Frame {frame_no}] 首帧数字: {current_number}（无前一帧） ✅")
            self.last_number = current_number
            return True

        if current_number >= self.last_number:
            if error_retry is False:
                if len(str(current_number)) > 7:
                    print(f"[Frame {frame_no}] 当前数字{current_number}长度不对,去除置信度最低的")
                    new_detections,new_current_number = self.truncate_and_get_number(detections)
                    detections.clear()
                    detections.extend(new_detections)
                    current_number = new_current_number
            print(f"[Frame {frame_no}] 当前数字{current_number} >= 上一帧数字{self.last_number} ✅递增")
            self.last_number = current_number
            return True
        else:
            print(f"[Frame {frame_no}] 当前数字{current_number} < 上一帧数字{self.last_number} ❌未递增")
            # 错误重试检测时,不更新错误帧数据
            if error_retry is False:
                # 记录错误帧及其前两帧
                self.error_buffer = list(self.history)[-self.retry_frame_size:]  # 包括当前帧和前两帧

            self.last_number = current_number  # 仍然更新为当前值，保持和流程一致
            return False
