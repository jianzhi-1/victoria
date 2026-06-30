from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class KvCache:
    """
    Expects K, V to be of shape (..., S, d)
    """
    _K: torch.Tensor | None = None
    _V: torch.Tensor | None = None
    S: int = 0
    """
    Valid sequence length.
    NOTE: This is needed since K, V can be pre-allocated.
    """

    def __post_init__(self) -> None:
        if self.K is None or self.V is None:
            assert self.K is None and self.V is None, [self.K, self.V]
            return
        assert self.K.shape == self.V.shape, [self.K.shape, self.V.shape]

    def empty(self) -> bool:
        return self.K is None or self.V is None

    @property
    def allocated_size(self) -> int:
        if self._K is None: return 0
        return self._K.shape[-2]
    
    @property
    def K(self) -> torch.Tensor | None:
        if self._K is None: return None
        return self._K[...,:self.S,:]

    @property
    def V(self) -> torch.Tensor | None:
        if self._V is None: return None
        return self._V[...,:self.S,:]
    
    def check_compatible(self, x: torch.Tensor) -> None:
        if self.K is None: return
        assert x.shape[:-2] == self.K.shape[:-2] and x.shape[-1] == self.K.shape[-1], [x.shape, self.K.shape]

    def append(self, K: torch.Tensor, V: torch.Tensor) -> None:
        assert K.shape == V.shape, [K.shape, V.shape]
        self.check_compatible(K)
        self.check_compatible(V)

        S_initial = self.S
        self.S += K.shape[-2]
        if self._K is None or self._V is None:
            self._K = K
            self._V = V

        if self.S <= self.allocated_size:
            self._K[...,S_initial:self.S,:] = K
            self._V[...,S_initial:self.S,:] = V
        else:
            self._K = torch.cat([self._K[...,:S_initial,:], K], dim=-2)
            self._V = torch.cat([self._V[...,:S_initial,:], V], dim=-2)

if __name__ == "__main__":
    # TODO: please write better tests.
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

        
