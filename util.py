import yaml
import numpy as np
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
    
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj