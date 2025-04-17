import asyncio
from track import Tracker
#import time
from client import wsClient

from util import load_config


async def main(server, client_id, detect_time):

    tracker = Tracker(
        model_path="yolov8n.pt", 
        stable_frames_threshold=48,
        verbose=False,  # 这里设置为False来禁止YOLO输出
        tracker="sort"
    )

    video_source = "traffic.avi"  # 或者使用摄像头：video_source = 0
    tracking_task = asyncio.create_task(tracker.track_objects(video_source))
    client = wsClient(server, client_id, detect_time, tracker)
    client_task = asyncio.create_task(client.client_control())
    data_collect_task = asyncio.create_task(client.data_collector())
    await asyncio.gather(tracking_task, client_task, data_collect_task)


if __name__ == "__main__":
    try:
        server, client_id, detect_time = load_config()
        print(f"Server: {server}")
        print(f"Client ID: {client_id}")
        print(f"Detect Time: {detect_time}")
    except Exception as e:
        print(f"加载配置失败: {e}")

    asyncio.run(main(server, client_id, detect_time))