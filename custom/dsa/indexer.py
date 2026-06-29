import torch
import torch.nn as nn
import torch.nn.functional as F

class DSAIndexer(nn.Module):
    def __init__(self, N: int, D: int, H: int, activation_type: type[nn.Module]) -> None:
        super().__init__()
        self.D = D
        self.N = N
        self.H = H
        self.Wk = nn.Linear(D, H)
        self.Wq = nn.Linear(D, N*H)
        self.Ww = nn.Linear(D, N)
        self.activation = activation_type()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]
        W = self.Ww(x)
        K = self.Wk(x)
        Q = self.Wq(x).view(B, S, self.N, self.H)
        QK = self.activation(torch.einsum("bsnh,bth->bnst", Q, K))
        mask = torch.tril(torch.ones(size=(S, S), device=x.device), diagonal=0)
        out = torch.einsum("bsn,bnst->bst", W, QK * mask)
        out_mask = torch.triu(torch.ones(size=(S, S), device=x.device), diagonal=1)
        out = out.masked_fill(out_mask.bool(), float("-inf"))
        return out
    
if __name__ == "__main__":
    B = 16
    S = 256
    D = 32
    N = 4
    H = 8
    net = DSAIndexer(
        N=N,
        D=D,
        H=H,
        activation_type=nn.ReLU
    )
    x = torch.randn(B, S, D)
    print(net(x))

