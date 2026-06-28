import numpy as np
from typing import Sequence

class Accumulator:
    def __init__(self):
        self.storage = []
        self.sorted = False
        self.rs = 0.
        self.w = 0
        # TODO: use w in quantile computation as well.

    def push(self, x: float, w: int = 1) -> None:
        self.storage.append(x)
        self.rs += x
        self.w += w
    
    def clear(self) -> None:
        self.storage.clear()

    def sort(self) -> None:
        if self.sorted: return
        self.storage.sort()

    def quantile(self, p: float | Sequence[float]) -> tuple[float, ...]:
        self.sort()
        if isinstance(p, (float, int)):
            assert 0 <= p <= 1, p
        else:
            for pp in p: assert 0 <= pp <= 1, pp
        return np.quantile(self.storage, p)
    
    def median(self) -> float:
        med, = self.quantile(0.5)
        return med
    
    def mean(self) -> float:
        return self.rs / self.w

    def max(self) -> float:
        self.sort()
        return self.storage[-1]

    def min(self) -> float:
        self.sort()
        return self.storage[0]




