import asyncio
from threading import Thread
import aiohttp


class AsyncClient:
    def __init__(self):
        self.session = None
        self.queue = asyncio.PriorityQueue()
        self.is_running = False

    async def start(self):
        self.session = aiohttp.ClientSession()
        self.is_running = True
        asyncio.create_task(self.process_queue())

    async def restart_session(self):
        await self.session.close()
        self.session = aiohttp.ClientSession()

    async def stop(self):
        self.is_running = False
        await self.queue.join()  # Wait for all tasks to complete
        await self.session.close()

    async def make_request(self, id, data, callback, put_first=False):
        priority = 0 if put_first else 1
        await self.queue.put((priority, (id, data, callback)))

    async def process_queue(self):
        while self.is_running:
            try:
                _, (id, data, callback) = await self.queue.get()
                asyncio.create_task(self.handle_request(id, data, callback))
            except Exception as e:
                print(f"Error in process_queue: {e}")

    async def handle_request(self, id, data, callback):
        raise NotImplementedError("handle_request must be implemented in a subclass")


class SyncClient:
    def __init__(self, async_client):
        self.async_client = async_client
        self.stop_event = asyncio.Event()
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.async_client.start())
        while not self.stop_event.is_set():
            self.loop.run_until_complete(asyncio.sleep(0.1))
        self.loop.run_until_complete(self.async_client.stop())
        self.loop.close()

    def make_request(self, id, data, callback):
        if self.stop_event.is_set():
            raise Exception("Client is stopped")

        async def _make_request():
            await self.async_client.make_request(id, data, callback)

        future = asyncio.run_coroutine_threadsafe(_make_request(), self.loop)
        future.result()

    def stop(self):
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                print("Thread is still alive")
