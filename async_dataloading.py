import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset


# 📘 Create a synthetic dataset for linear regression
class SyntheticDataset(Dataset):
    def __init__(self, a, b, noise_std, num_samples):
        self.a = a
        self.b = b
        self.noise_std = noise_std
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(3, 224, 224)
        noise = torch.randn(3, 224, 224) * self.noise_std
        y = self.a * x + self.b + noise
        return x, y


# 📘 Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.in_linear = nn.Linear(
            in_features=3 * 224 * 224, out_features=1024
        )
        self.middle_linear = nn.Linear(in_features=1024, out_features=1024)
        self.out_linear = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        out = F.gelu(self.in_linear(x.view(x.shape[0], -1)))
        out = F.gelu(self.middle_linear(out))
        out = self.out_linear(out)
        return out


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


# 📘 Training loop
async def train_model(
    async_data_generator, model, criterion, optimizer, num_epochs, length
):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # 📘 Run the AsyncGeneratorWrapper and get the sync generator
        async_dataloading_wrapper = AsyncGeneratorWrapper(data_loaders)
        await async_dataloading_wrapper.start()

        with tqdm.tqdm(total=len(async_data_generator)) as pbar:
            async for inputs, targets in async_dataloading_wrapper.process_queue():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")


# 📘 Create DataLoader instances
datasets = [
    SyntheticDataset(a=2, b=3, noise_std=0.1, num_samples=100),
    SyntheticDataset(a=2, b=3, noise_std=0.1, num_samples=200),
    SyntheticDataset(a=2, b=3, noise_std=0.1, num_samples=300),
]

data_loaders = [
    DataLoader(dataset, batch_size=1, shuffle=True) for dataset in datasets
]

# 📘 Create the AsyncGeneratorWrapper instance
async_data_generator = AsyncGeneratorWrapper(data_loaders)

# 📘 Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 📘 Train the model
num_epochs = 10
asyncio.run(
    train_model(
        async_data_generator,
        model,
        criterion,
        optimizer,
        num_epochs,
        len(async_data_generator),
    )
)
