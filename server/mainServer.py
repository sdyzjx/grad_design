import time
import asyncio
from wsServer import wsSocket



async def main():
    ws_socket = wsSocket(4000)
    ws_task = asyncio.create_task(ws_socket.start_ws_server())
    await ws_task


if __name__ == '__main__':
    asyncio.run(main())