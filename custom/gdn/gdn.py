import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, D: int, DV: int, DK: int) -> None:
        super().__init__()
        self.D = D
        self.DV = DV
        self.DK = DK
        self.Wa = nn.Linear(D, 1)
        self.Wb = nn.Linear(D, 1)
        self.Wq = nn.Linear(D, DK)
        self.Wk = nn.Linear(D, DK)
        self.Wv = nn.Linear(D, DV)

    def forward(self, x: torch.Tensor, prev_S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]
        assert prev_S.shape == (B, self.DV, self.DK), [prev_S.shape, (B, self.DV, self.DK)]
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        alpha = torch.sigmoid(self.Wa(x))
        beta = torch.sigmoid(self.Wb(x))
        S = torch.einsum("bc,bvk->bvk", alpha, prev_S) - torch.einsum("bc,bvk->bvk", alpha * beta, (torch.einsum("bv,bk->bvk", torch.einsum("bvk,bk->bv", prev_S, K), K))) + torch.einsum("bc,bvk->bvk", beta, torch.einsum("bv,bk->bvk", V, K))
        out = torch.einsum("bvk,bk->bv", S, Q)
        return out, S

if __name__ == "__main__":
    B = 16
    DV = 32
    D = 8
    DK = 16
    gdn = GDN(D, DV, DK)
    S = torch.zeros(B, DV, DK, requires_grad=False)
    L = 10
    for _ in range(L):
        x = torch.randn(size=(B, D))
        out, S = gdn(x, S)
        print(out)
    print("DONE")