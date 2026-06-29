import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self, D: int, H: int, O: int) -> None:
        super().__init__()
        self.D = D
        self.net = nn.Sequential(
            nn.Linear(D, H),
            nn.SiLU(),
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, O)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]
        return self.net(x)
