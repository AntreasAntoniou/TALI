# 📘 Define the AsyncGeneratorWrapper class
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import queue
import threading
from typing import List
from torch.utils.data import DataLoader


# 📘 Define the AsyncGeneratorWrapper class
class AsyncGeneratorWrapper:
    def __init__(self, data_loaders: List[DataLoader]):
        self.data_loaders = data_loaders
        self.queue = asyncio.Queue()

    def wrapper(self, data_loader):
        for value in data_loader:
            self.queue.put_nowait(value)
        self.queue.put_nowait(None)

    async def process_queue(self):
        num_none_received = 0
        while num_none_received < len(self.data_loaders):
            value = await self.queue.get()
            if value is None:
                num_none_received += 1
            else:
                yield value

    async def run(self):
        with ThreadPoolExecutor() as executor:
            tasks = [
                executor.submit(self.wrapper, dl) for dl in self.data_loaders
            ]
            await asyncio.gather(
                *[asyncio.wrap_future(future) for future in tasks]
            )

    async def start(self):
        return asyncio.ensure_future(self.run())

    def __len__(self):
        return sum(len(dl) for dl in self.data_loaders)
