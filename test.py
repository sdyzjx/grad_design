import cv2
import torch
from ultralytics import YOLO
from sort.sort import Sort  # 导入本地的SORT实现
# 初始化YOLO模型和SORT追踪器
model = YOLO("yolov8n.pt")
tracker = Sort()
# 打开视频流（可以是视频文件路径或摄像头索引）
video_source = "traffic.avi"  # 或者使用摄像头：video_source = 0
cap = cv2.VideoCapture(video_source)
# 用于存储每个对象的ID及其出现的帧数
tracked_objects_history = {}
# 设置稳定帧数的阈值
stable_frames_threshold = 48
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 使用YOLO模型进行预测
    results = model(frame)
    # 提取检测框信息
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist() + [box.conf[0].item(), box.cls[0].item()]
            detections.append([x1, y1, x2, y2, conf])
    # 更新SORT追踪器
    tracked_objects = tracker.update(torch.tensor(detections))
    # 更新每个对象的ID及其出现的帧数
    current_frame_ids = set()
    for obj in tracked_objects:
        obj_id = int(obj[4])
        current_frame_ids.add(obj_id)
        if obj_id in tracked_objects_history:
            tracked_objects_history[obj_id] += 1
        else:
            tracked_objects_history[obj_id] = 1
    # 移除在当前帧中未出现的对象
    for obj_id in list(tracked_objects_history.keys()):
        if obj_id not in current_frame_ids:
            del tracked_objects_history[obj_id]
    # 绘制追踪结果（只有当对象的ID在连续几帧中保持不变时才绘制）
    for obj in tracked_objects:
        obj_id = int(obj[4])
        if tracked_objects_history.get(obj_id, 0) >= stable_frames_threshold:
            x1, y1, x2, y2 = obj[:4].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 显示结果
    cv2.imshow("YOLOv8 + SORT", frame)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# 释放资源
cap.release()
cv2.destroyAllWindows()