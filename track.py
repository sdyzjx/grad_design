import cv2
import torch
import asyncio
from ultralytics import YOLO
from sort.sort import Sort

class AsyncObjectTracker:
    def __init__(self, model_path="yolov8n.pt", stable_frames_threshold=48, verbose=False):
        """
        初始化异步对象追踪器
        
        参数:
            model_path: YOLO模型路径
            stable_frames_threshold: 稳定帧数阈值
            verbose: 是否显示详细输出
        """
        self.model = YOLO(model_path)
        self.tracker = Sort()
        self.stable_frames_threshold = stable_frames_threshold
        self.tracked_objects_history = {}
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
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist() + [box.conf[0].item(), box.cls[0].item()]
                detections.append([x1, y1, x2, y2, conf])
        
        # 更新SORT追踪器
        tracked_objects = self.tracker.update(torch.tensor(detections))
        
        # 更新每个对象的ID及其出现的帧数
        current_frame_ids = set()
        for obj in tracked_objects:
            obj_id = int(obj[4])
            current_frame_ids.add(obj_id)
            if obj_id in self.tracked_objects_history:
                self.tracked_objects_history[obj_id] += 1
            else:
                self.tracked_objects_history[obj_id] = 1
        
        # 移除在当前帧中未出现的对象
        for obj_id in list(self.tracked_objects_history.keys()):
            if obj_id not in current_frame_ids:
                del self.tracked_objects_history[obj_id]
        
        # 存储当前检测结果
        self.current_detections = []
        for obj in tracked_objects:
            obj_id = int(obj[4])
            if self.tracked_objects_history.get(obj_id, 0) >= self.stable_frames_threshold:
                x1, y1, x2, y2 = obj[:4].astype(int)
                self.current_detections.append({
                    'id': obj_id,
                    'bbox': (x1, y1, x2, y2),
                    'age': self.tracked_objects_history[obj_id],
                    'class': self.model.names[int(obj[5])] if len(obj) > 5 else 'unknown'
                })
                
                # 绘制追踪结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), 
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