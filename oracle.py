import torch
import torch.nn as nn
import h5py
from pathlib import Path

class OracleNet(nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__()
        self.D = D
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.Tanh(),
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]
        return self.net(x)
    

class OracleDataset:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def generate(self, path_prefix: str, N: int, n: int, D: int) -> None:
        oracle_net = OracleNet(D)
        torch.manual_seed(self.seed)
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        for shard_idx, i in enumerate(range(0, N, n)):
            B = n if i + n < N else N - i
            X = torch.randn((B, D))
            with torch.no_grad():
                y = oracle_net(X)
            with h5py.File(f"{path_prefix}_{shard_idx}.h5", "w") as f:
                f.create_dataset("X", data=X.cpu().numpy())
                f.create_dataset("y", data=y.cpu().numpy())
        
if __name__ == "__main__":
    oracle = OracleDataset(42)
    oracle.generate("./data/data", 100, 16, 16)
