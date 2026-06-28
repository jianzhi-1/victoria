import torch
import h5py
import glob
from typing import Iterator
from pathlib import Path
import torch.distributed as dist

class StreamingDataset[T](torch.utils.data.IterableDataset):
    def __init__(self, path_prefix: str, debug: bool = False) -> None:
        super().__init__()
        self.path_prefix = path_prefix
        if debug:
            print(f"found {len(glob.glob(f"{path_prefix}*"))} shards")

    def __iter__(self) -> Iterator[T]:
        worker_info = torch.utils.data.get_worker_info()

        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        cur = rank * num_workers + worker_id
        delta = world_size * num_workers

        data_path = f"{self.path_prefix}_{cur}.h5"
        CHUNK_SIZE = 1024
        while Path(data_path).exists():
            with h5py.File(data_path, "r") as f:
                X, y = f["X"], f["y"]
                assert len(X) == len(y), [len(X), len(y)]
                for chunk_start in range(0, len(X), CHUNK_SIZE):
                    chunk_end = min(len(X), chunk_start + CHUNK_SIZE)
                    X_chunk = X[chunk_start:chunk_end]
                    y_chunk = y[chunk_start:chunk_end]
                    for Xp, yp in zip(X_chunk, y_chunk, strict=True):
                        yield Xp, yp
            cur += delta
            data_path = f"{self.path_prefix}_{cur}.h5"

if __name__ == "__main__":
    B = 16
    dataset = StreamingDataset(path_prefix="./data/data")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=B)
    for X, y in dataloader:
        print(X.shape, y.shape)