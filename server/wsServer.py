import websockets
import asyncio
import json
from websockets.legacy.server import WebSocketServerProtocol

class wsSocket:
    client = {}
    test_state = False

    def __init__(self, _port):
        self.port = _port
        print("Websocket server started at port " + str(self.port) + " on 0.0.0.0")
        self.clients = {}

    async def ws_handle(self, websocket: WebSocketServerProtocol):
        async for message in websocket:
            print(message)
            json_recv = json.loads(message)
            if "client_id" in json_recv:
                id = json_recv["client_id"]
                action = json_recv["action"]
                if id not in self.clients and action == "100":
                    self.clients[id] = websocket
                    print("New connection " + id + " established successfully")
                    await self.sendmsg(id, "100_1")
                if id in self.clients and action == "800":
                    await self.sendmsg(id, "800_1")
                    del self.clients[id]
                    print(id + " disconnected")
                if action == "600":
                    self.clients[id] = websocket
                    await self.sendmsg(id, "600_1")
                    print(id + " reconnected")


    async def sendmsg(self, client_id, msg):
        # if(websocket in clients):
        msg_send = {
            "msg": msg
        }
        json_data = json.dumps(msg_send)
        websocket = self.clients[client_id]
        print("Sending " + msg + " to " + client_id)
        await websocket.send(json_data)

    async def start_ws_server(self):
        async with websockets.serve(self.ws_handle, "0.0.0.0", self.port):
            await asyncio.Future()
