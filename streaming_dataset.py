import torch
import h5py
import glob
from typing import Iterator
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import IterableDataset
import numpy as np

class StreamingDataset[T](torch.utils.data.IterableDataset):
    CHUNK_SIZE = 1024

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
        while Path(data_path).exists():
            with h5py.File(data_path, "r") as f:
                X, y = f["X"], f["y"]
                assert len(X) == len(y), [len(X), len(y)]
                for chunk_start in range(0, len(X), self.CHUNK_SIZE):
                    chunk_end = min(len(X), chunk_start + self.CHUNK_SIZE)
                    X_chunk = X[chunk_start:chunk_end]
                    y_chunk = y[chunk_start:chunk_end]
                    for Xp, yp in zip(X_chunk, y_chunk, strict=True):
                        yield Xp, yp
            cur += delta
            data_path = f"{self.path_prefix}_{cur}.h5"

class SplitStreamingDataset[T](torch.utils.data.IterableDataset):
    EPS = 1e-8

    def __init__(self, source: torch.utils.data.IterableDataset, ratios: dict[str, float], sample_type: str, seed: int) -> None:
        super().__init__()
        assert sample_type in ratios, [sample_type, ratios]
        assert abs(sum(ratios.values()) - 1.) < self.EPS, sum(ratios.values())
        self.source = source
        self.ratios = ratios
        self.sample_type = sample_type
        self.lb, self.ub = 0., ratios[sample_type]
        for k, v in ratios.items():
            if k == sample_type: break
            self.lb += v
            self.ub += v
    
    def __iter__(self) -> Iterator[T]:
        rng = np.random.default_rng(seed=seed)
        for data in self.source:
            if self.lb <= rng.random() < self.ub:
                yield data

def get_splits(dataset: StreamingDataset, ratios: dict[str, float], seed: int) -> tuple[SplitStreamingDataset, ...]:
    # NOTE: purposely chose the design where datasets are streamed once for each stream.
    return tuple(SplitStreamingDataset(dataset, ratios, sample_type, seed) for sample_type in ratios.keys())

if __name__ == "__main__":
    B = 16
    dataset = StreamingDataset(path_prefix="./data/data")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=B)
    # note: shuffle cannot be True
    for X, y in dataloader:
        print(X.shape, y.shape)

    ratios = {
        "train": 0.7,
        "eval": 0.3
    }
    seed = 42
    train_dataset, eval_dataset = get_splits(dataset, ratios, seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=B)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=B)

    print("train")
    for X, y in train_dataloader:
        print(X.shape, y.shape)

    print("eval")
    for X, y in eval_dataloader:
        print(X.shape, y.shape)