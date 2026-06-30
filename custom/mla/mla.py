import torch
import torch.nn as nn
import torch.nn.functional as F

class MLA(nn.Module):
    """
    Key idea: compress K and V into R^Dc, decompress when needed.
    KV cache memory scales O(Dc * S).

    NOTE: N, d are independent of D, Dc.
    """
    def __init__(self, D: int, Dc: int, N: int, d: int) -> None:
        super().__init__()
        self.N = N
        self.D = D
        assert Dc < D, [Dc, D]
        self.d = d
        self.Dc = Dc
        self.Wc = nn.Linear(D, Dc)
        self.Wuk = nn.Linear(Dc, N * d)
        self.Wuv = nn.Linear(Dc, N * d)
        self.Wq = nn.Linear(D, N * d)
        self.Wo = nn.Linear(N * d, D)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement RoPE trick
        # TODO: implement Wq <-> Wuk absorption trick
        # TODO: implement query compression
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]
        c = self.Wc(x)
        K, V = self.Wuk(c).view(B, S, self.N, self.d).transpose(1, 2).contiguous(), self.Wuv(c).view(B, S, self.N, self.d).transpose(1, 2).contiguous()
        Q = self.Wq(x).view(B, S, self.N, self.d).transpose(1, 2).contiguous()
        qkt = torch.einsum("bnsd,bntd->bnst", Q, K) / (self.d ** 0.5)
        alpha = F.softmax(qkt, dim=-1)
        att = torch.einsum("bnst,bntd->bnsd", alpha, V).transpose(1, 2).contiguous().view(B, S, self.N * self.d)
        out = self.Wo(att)
        return out
    
if __name__ == "__main__":
    B = 16
    D = 32
    Dc = 8
    N = 4
    d = 4
    S = 512

    net = MLA(D, Dc, N, d)
    x = torch.randn(size=(B, S, D))
    print(net(x).shape)
