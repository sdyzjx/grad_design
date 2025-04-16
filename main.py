import asyncio
from track import Tracker
#import time
from client import wsClient
import yaml

def load_config(config_path='config.yaml'):
    """
    从YAML配置文件中读取配置项
    
    参数:
        config_path (str): 配置文件路径，默认为'config.yaml'
    
    返回:
        tuple: (server, client_id, detect_time)
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
            server = config.get('server')
            client_id = config.get('client_id')
            detect_time = config.get('detect_time')
            
            # 验证必要配置是否存在
            if None in (server, client_id, detect_time):
                raise ValueError("配置文件中缺少必要的配置项")
                
            return server, client_id, detect_time
            
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 未找到")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML解析错误: {e}")
    

async def main(server, client_id, detect_time):
    # 初始化异步对象追踪器，设置verbose=False来禁止YOLO输出
    tracker = Tracker(
        model_path="yolov8n.pt", 
        stable_frames_threshold=48,
        verbose=False,  # 这里设置为False来禁止YOLO输出
        tracker="sort"
    )

    # 启动追踪任务
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