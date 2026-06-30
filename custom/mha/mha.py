import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, D: int, H: int, N: int) -> None:
        super().__init__()
        self.D = D
        self.H = H
        self.N = N
        assert H % N == 0, [H, N]
        self.d = H // N
        self.Wk = nn.Linear(D, H)
        self.Wq = nn.Linear(D, H)
        self.Wv = nn.Linear(D, H)
        self.Wo = nn.Linear(H, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: add masking
        # TODO: add decode mechanisms
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]
        Q = self.Wq(x).view(B, S, self.N, self.d).transpose(1, 2)
        K = self.Wk(x).view(B, S, self.N, self.d).transpose(1, 2)
        V = self.Wv(x).view(B, S, self.N, self.d).transpose(1, 2)

        qkt = torch.einsum("bnsd,bntd->bnst", Q, K) / (self.d ** 0.5)
        alpha = F.softmax(qkt, dim=-1)
        att = torch.einsum("bnst,bntd->bsnd", alpha, V).contiguous().view(B, S, self.H)
        out = self.Wo(att)
        assert out.shape == (B, S, self.D), [out.shape, (B, S, self.D)]
        return out
    
if __name__ == "__main__":
    torch.manual_seed(42)
    B = 16
    D = 256
    H = 512
    N = 4
    S = 1024
    x = torch.randn(size=(B, S, D))
    net = MHA(D, H, N)
    print(net(x).shape)