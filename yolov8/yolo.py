import cv2
import torch
from ultralytics import YOLO


class Yolov8Manager:
    def __init__(self, model_path, task="obb", device="cuda"):
        """
        初始化 YOLOv8 模型

        :param model_path: 模型权重路径，如 'best.pt'
        :param task: 任务类型，如 'obb' (定向边界框)
        :param device: 使用设备，'cuda' 或 'cpu'
        """
        self.model = YOLO(model_path, task=task)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.names = self.model.names  # 获取类别名称列表
        print(f"[INFO] Model loaded on device: {self.device}")

    def predict(self, image_source, conf=0.75, iou=0.75, imgsz=640,
                save=False, project="output", name="prediction", exist_ok=True):
        """
        对输入图像进行预测

        :param image_source: 图像路径(str) 或 numpy.ndarray
        :param conf: 置信度阈值
        :param iou: NMS IoU 阈值
        :param imgsz: 输入图像尺寸
        :param save: 是否保存可视化结果
        :param project: 输出目录
        :param name: 保存的文件名前缀
        :param exist_ok: 是否允许覆盖已有目录
        :return: {
            'image': 原始图像,
            'results': 检测结果列表 [
                {
                    'points': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                    'class_name': 类别名,
                    'class_id': 类别ID,
                    'confidence': 置信度,
                    'angle': 旋转角度（弧度）
                }, ...
            ]
        }
        """
        # 加载图像
        if isinstance(image_source, str):
            original_image = cv2.imread(image_source)
            source = image_source
        elif isinstance(image_source, np.ndarray):
            original_image = image_source.copy()
            source = original_image
        else:
            raise ValueError("image_source must be a path string or a numpy array.")

        # 推理
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            save=save,
            project=project,
            name=name,
            exist_ok=exist_ok,
            save_txt=False,
            save_conf=True,
            save_crop=False,
            show=False
        )

        parsed_results = []
        result = results[0]  # 取第一个结果对象

        if hasattr(result, 'obb') and result.obb is not None:
            obb = result.obb
            for i in range(len(obb)):
                pts = obb.xyxyxyxy[i]
                cls_idx = obb.cls[i]
                conf_score = obb.conf[i]

                points = pts.cpu().numpy().tolist()
                class_id = int(cls_idx.item())
                class_name = self.names[class_id]
                confidence = float(conf_score.item())

                # 提取旋转角度（弧度）
                angle = obb.xywhr[i][4].cpu().item() if len(obb.xywhr) > 0 else None

                parsed_results.append({
                    'points': points,
                    'class_name': class_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'angle': angle
                })

        return {
            'image': original_image,
            'results': parsed_results
        }

# 示例使用：
if __name__ == "__main__":
    model_path = "/home/bz/workspace/ultralytics/runs/obb/yolov8_obb5/weights/best.pt"
    yolo_manager = Yolov8Manager(model_path, device="cuda")

    image_path = "/home/bz/workspace/ultralytics/yolov8/datasets/yolov8-reiver-obb/test/images/1c848aa7-1051434267537.jpg"
    output = yolo_manager.predict(image_path, conf=0.75, save=True, project="inference_output", name="result_1")

    # 打印结果
    for res in output['results']:
        print(res)

    # 显示图像
    cv2.imshow("Detected Image", output['image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()