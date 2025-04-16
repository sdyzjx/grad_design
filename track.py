import cv2
import torch
import asyncio
from ultralytics import YOLO
from sort import Sort
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, model_path="yolov8n.pt", stable_frames_threshold=48, verbose=False, tracker="sort"):
        """
        初始化异步对象追踪器
        
        参数:
            model_path: YOLO模型路径
            stable_frames_threshold: 稳定帧数阈值
            verbose: 是否显示详细输出
            tracker: 选择追踪器类型 ("sort" 或 "deep_sort")
        """
        self.tracker_choice = tracker
        self.model = YOLO(model_path)
        if tracker == "sort":
            self.tracker = Sort()
        elif tracker == "deep_sort":
            self.tracker = DeepSort()
        
        self.stable_frames_threshold = stable_frames_threshold
        self.tracked_objects_history = {}  # 存储追踪对象的历史信息
        self.tracked_objects_classes = {}  # 新增：存储追踪对象的类别信息
        self.current_detections = []
        self.running = False
        self.verbose = verbose
        
    async def process_frame(self, frame):
        """
        异步处理单个帧，执行检测和追踪
        
        参数:
            frame: 输入视频帧
            
        返回:
            处理后的帧和检测结果
        """
        # 使用YOLO模型进行预测，设置verbose=False来禁止输出
        results = self.model(frame, verbose=self.verbose)
        
        # 提取检测框信息
        detections = []
        detections_deepsort = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist() + [box.conf[0].item(), box.cls[0].item()]
                detections.append([x1, y1, x2, y2, conf])
                w, h = x2 - x1, y2 - y1
                class_name = self.model.names[int(cls)]
                detections_deepsort.append(([x1, y1, w, h], conf, class_name))
        
        # 更新追踪器
        if self.tracker_choice == "sort":
            # 对于SORT，我们需要在更新追踪器前保存检测结果的类别信息
            tracked_objects = self.tracker.update(torch.tensor(detections))
            
            # 创建一个字典来映射检测框到类别
            detection_classes = {}
            for det, box in zip(detections, results[0].boxes):
                cls = int(box.cls[0].item())
                detection_classes[tuple(det[:4])] = cls
            
        elif self.tracker_choice == "deep_sort":
            tracked_objects = self.tracker.update_tracks(detections_deepsort, frame=frame)
        
        # 更新每个对象的ID及其出现的帧数
        current_frame_ids = set()
        for obj in tracked_objects:
            if self.tracker_choice == "sort":
                obj_id = int(obj[4])
                current_frame_ids.add(obj_id)
                
                # 获取当前追踪对象的边界框
                x1, y1, x2, y2 = obj[:4]
                
                # 找到最匹配的原始检测框以获取类别
                matched_class = None
                min_distance = float('inf')
                
                for det in detections:
                    det_x1, det_y1, det_x2, det_y2, _ = det
                    # 计算IOU或中心点距离
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    det_center_x = (det_x1 + det_x2) / 2
                    det_center_y = (det_y1 + det_y2) / 2
                    distance = ((center_x - det_center_x)**2 + (center_y - det_center_y)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        matched_class = detection_classes.get((det_x1, det_y1, det_x2, det_y2), -1)
                
                # 更新或存储类别信息
                if obj_id in self.tracked_objects_history:
                    self.tracked_objects_history[obj_id] += 1
                    # 只有当之前没有类别信息或新检测到的类别可信时才更新
                    if matched_class != -1 and (obj_id not in self.tracked_objects_classes or self.tracked_objects_history[obj_id] < 5):
                        self.tracked_objects_classes[obj_id] = matched_class
                else:
                    self.tracked_objects_history[obj_id] = 1
                    if matched_class != -1:
                        self.tracked_objects_classes[obj_id] = matched_class
                    
            elif self.tracker_choice == "deep_sort":
                obj_id = obj.track_id
                current_frame_ids.add(obj_id)
                if obj_id in self.tracked_objects_history:
                    self.tracked_objects_history[obj_id] += 1
                else:
                    self.tracked_objects_history[obj_id] = 1
        
        # 移除在当前帧中未出现的对象
        for obj_id in list(self.tracked_objects_history.keys()):
            if obj_id not in current_frame_ids:
                del self.tracked_objects_history[obj_id]
                if obj_id in self.tracked_objects_classes:
                    del self.tracked_objects_classes[obj_id]
        
        # 存储当前检测结果
        self.current_detections = []
        for obj in tracked_objects:
            if self.tracker_choice == "sort":
                obj_id = int(obj[4])
                if self.tracked_objects_history.get(obj_id, 0) >= self.stable_frames_threshold:
                    x1, y1, x2, y2 = obj[:4].astype(int)
                    class_id = self.tracked_objects_classes.get(obj_id, -1)
                    class_name = self.model.names[class_id] if class_id != -1 else 'unknown'
                    
                    self.current_detections.append({
                        'id': obj_id,
                        'bbox': (x1, y1, x2, y2),
                        'age': self.tracked_objects_history[obj_id],
                        'class': class_name
                    })
                    
                    # 绘制追踪结果
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {obj_id} {class_name}', (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            elif self.tracker_choice == "deep_sort":
                obj_id = obj.track_id
                if self.tracked_objects_history.get(obj_id, 0) >= self.stable_frames_threshold:
                    bbox = obj.to_ltrb()  # 获取边界框坐标 [left, top, right, bottom]
                    x1, y1, x2, y2 = map(int, bbox)
                    class_id = obj.get_det_class() if hasattr(obj, 'get_det_class') else -1
                    class_name = self.model.names[class_id] if class_id != -1 else 'unknown'
                    
                    self.current_detections.append({
                        'id': obj_id,
                        'bbox': (x1, y1, x2, y2),
                        'age': self.tracked_objects_history[obj_id],
                        'class': class_name
                    })
                    
                    # 绘制追踪结果
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {obj_id} {class_name}', (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

    async def track_objects(self, video_source):
        """
        异步追踪视频流中的对象
        
        参数:
            video_source: 视频源(文件路径或摄像头索引)
        """
        self.running = True
        cap = cv2.VideoCapture(video_source)
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 异步处理帧
            processed_frame = await self.process_frame(frame)
            
            # 显示结果
            cv2.imshow("Async Object Tracking", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            await asyncio.sleep(0.001)
            # 防止阻塞
        cap.release()
        cv2.destroyAllWindows()
        
    def stop_tracking(self):
        """停止追踪"""
        self.running = False
        
    def get_current_detections(self):
        """获取当前检测结果"""
        return self.current_detections