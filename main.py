import asyncio
from track import AsyncObjectTracker
import time

async def main():
    # 初始化异步对象追踪器，设置verbose=False来禁止YOLO输出
    tracker = AsyncObjectTracker(
        model_path="yolov8n.pt", 
        stable_frames_threshold=48,
        verbose=True,  # 这里设置为False来禁止YOLO输出
        tracker="deep_sort"
    )

    # 启动追踪任务
    video_source = "traffic.avi"  # 或者使用摄像头：video_source = 0
    tracking_task = asyncio.create_task(tracker.track_objects(video_source))
    await tracking_task
    # 设置运行时间为30秒
    start_time = time.time()

    try:
        while time.time() - start_time < 30:
            # 获取并打印当前检测结果
            detections = tracker.get_current_detections()
            print(f"\n当前帧检测到的对象({len(detections)}个):")
            for obj in detections:
                print(obj)
            
            # 每秒更新一次
            await asyncio.sleep(1)
    finally:
        # 停止追踪
        tracker.stop_tracking()
        await tracking_task



if __name__ == "__main__":
    asyncio.run(main())