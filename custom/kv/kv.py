from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class KvCache:
    """
    Expects K, V to be of shape (..., S, d)
    """
    K: torch.Tensor
    V: torch.Tensor
    S: int = 0
    """
    Valid sequence length.
    NOTE: This is needed since K, V can be pre-allocated.
    """

    def __post_init__(self) -> None:
        assert self.K.shape == self.V.shape, [self.K.shape, self.V.shape]

    @property
    def allocated_size(self) -> int:
        return self.K.shape[-2]

    def append(self, K: torch.Tensor, V: torch.Tensor) -> None:
        assert K.shape == V.shape, [K.shape, V.shape]
        S_initial = self.S
        self.S += K.shape[-2]
        if self.S <= self.allocated_size:
            self.K[...,S_initial:self.S,:] = K
            self.V[...,S_initial:self.S,:] = V
        else:
            self.K = torch.cat([self.K[...,:S_initial,:], K], dim=-2)
            self.V = torch.cat([self.V[...,:S_initial,:], V], dim=-2)

if __name__ == "__main__":
    B = 16
    MAX_SEQLEN = 1024
    D = 128
    K_prealloc = torch.zeros(size=(B, MAX_SEQLEN, D))
    V_prealloc = torch.zeros(size=(B, MAX_SEQLEN, D))
    kv = KvCache(K_prealloc, V_prealloc)

    def generate_kv(s: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(size=(B, s, D)), torch.randn(size=(B, s, D))

    for _ in range(16):
        k, v = generate_kv(128)
        kv.append(k, v)

        
