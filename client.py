'''
模块作用：当检测到确定目标时，把数据推送到云端服务器
输入：轮询检测结果时间间隔，websocket服务器地址

'''
from collections import deque
import asyncio
from datetime import *
import aioconsole
import websockets
import json
import util
class wsClient:
    def __init__(self, _address, _id, detect_time, tracker):
        self.address = _address
        self.client_id = _id
        self.shutdown_state = False
        self.connected = False
        self.retries = -1
        self.heartbeat = False
        self.detect_time = detect_time
        self.tracker = tracker
        self.detection_queue = deque([])
        # fetch initial server config
        # asyncio.run(self.client_start(mac, address, client_id))

    async def client_control(self):
        while True:
            # 发现没有连接进入重连模式
            if not self.connected:
                if self.retries == -1:
                    await self.client_start()
                else:
                    print("Strat reconnection")
                    await self.reconnect_server()

    async def client_start(self):
        async with websockets.connect(self.address) as self.ws:
            msg = {
                "client_id": self.client_id,
                "action": "100"
            }
            json_data = json.dumps(msg)
            await self.ws.send(json_data)
            response = await self.ws.recv()
            json_recv = json.loads(response)
            if "msg" in json_recv:
                msg = json_recv["msg"]
                if msg == "100_1":
                    print("服务器 " + self.address + " 连接成功.")
                    print("进入数据发送进程")
                    self.connected = True
                    self.retries = 0
                    await self.recv_send_handler()

    async def recv_send_handler(self):
        print("开始发送")
        while self.connected:
            try:
                '''
                                response = await self.ws.recv()
                json_recv = json.loads(response)
                if "msg" in json_recv:
                    msg = json_recv["msg"]
                    if msg == "800_1":
                        await self.ws.close()
                        self.shutdown_state = True
                        exit()
                    #这里写服务器端发送过来的控制命令

                        elif msg == "600":
                        msg_re = {
                            "id": self.client_id,
                            "action": "600_1"
                        }
                        json_data = json.dumps(msg_re)
                        await self.ws.send(json_data)

                '''
                await asyncio.sleep(1)
                if len(self.detection_queue) != 0:
                    data = self.detection_queue.popleft()
                    print(data)
                    obj_converted = util.convert_numpy_types(data)
                    obj_json = json.dumps(obj_converted)
                    await self.ws.send(obj_json)

            except Exception as e:
                print(e)
                self.connected = False
    #这个协程要记得在主函数中创建任务
    #使用队列的思想，缓存检测结果。如果发生网络中断，则在此过程中检测到的目标不会受到
    #影响
    async def data_collector(self):
        while True:
            detections = self.tracker.get_current_detections()
            if len(detections) != 0:
                #print(f"\n当前帧检测到的对象({len(detections)}个):")
                #print(f"\n当前帧检测到的不重复对象({len(self.detection_queue)}个):")
                for obj in detections:
                    if len(self.detection_queue) == 0:
                        self.detection_queue.append(obj)
                    # 检查当前对象的ID是否已经在队列中存在
                    if not any(existing_obj['id'] == obj['id'] for existing_obj in self.detection_queue):
                        self.detection_queue.append(obj)
            await asyncio.sleep(self.detect_time)

    async def reconnect_server(self):
        try:
            async with websockets.connect(self.address) as ws_t:
                msg = {
                    "id": self.client_id,
                    "action": "600"
                }
                json_data = json.dumps(msg)
                await ws_t.send(json_data)
                response = await ws_t.recv()
                json_recv = json.loads(response)
                if "msg" in json_recv:
                    msg = json_recv["msg"]
                    if msg == "600_1":
                        print("Server " + self.address + " reconnected succefully.")
                        self.connected = True
                        self.retries = 0
                        self.ws = ws_t
                        await self.recv_handler()
        except Exception as e:
            print("Reconnection failed, tried " + str(self.retries) + " times")
            self.retries = self.retries + 1

    async def client_stop(self):
        msg = {
            "id": self.client_id,
            "action": "800"
        }
        json_data = json.dumps(msg)
        await self.ws.send(json_data)

    async def get_input(self, prompt):
        user_input = await aioconsole.ainput(prompt)
        return user_input

    async def input_control(self):
        print(
            "Please choose action:\n" +
            "--------------------------------\n" +
            "s: Shutdown the client\n" +
            "m: Reconfigure mac address\n" +
            "--------------------------------"
        )
        while True:
            user_input = await self.get_input("Enter command:")
            if user_input == 's':
                await self.client_stop()
                print("Client has stopped")
                # exit()
            elif user_input == 'm':
                mac_input = await self.get_input("Enter new mac:")
                self.mac = mac_input
            if self.shutdown_state:
                exit()