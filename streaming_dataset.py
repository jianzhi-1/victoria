import torch
import h5py
import glob
from typing import Iterator
from pathlib import Path

class StreamingDataset[T](torch.utils.data.IterableDataset):
    def __init__(self, path_prefix: str, debug: bool = False) -> None:
        super().__init__()
        self.path_prefix = path_prefix
        if debug:
            print(f"found {len(glob.glob(f"{path_prefix}*"))} shards")

    def __iter__(self) -> Iterator[T]:
        worker_info = torch.utils.data.get_worker_info()
        cur = 0 if worker_info is None else worker_info.id
        delta = 1 if worker_info is None else worker_info.num_workers
        data_path = f"{self.path_prefix}_{cur}.h5"
        while Path(data_path).exists():
            with h5py.File(data_path, "r") as f:
                yield from zip(f["X"][:], f["y"][:], strict=True)
            cur += delta
            data_path = f"{self.path_prefix}_{cur}.h5"

if __name__ == "__main__":

    dataset = StreamingDataset(path_prefix="./data/data")
    print(torch.utils.data.DataLoader(dataset, num_workers=2))
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
    for X, y in dataloader:
        print(X, y)