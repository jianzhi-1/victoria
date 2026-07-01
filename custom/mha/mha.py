import torch
import torch.nn as nn
import torch.nn.functional as F
from custom.kv.kv import KvCache

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

    def forward(self, x: torch.Tensor, kv_cache: KvCache | None = None) -> torch.Tensor:
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]

        Q = self.Wq(x).view(B, S, self.N, self.d).transpose(1, 2)

        preK = self.Wk(x).view(B, S, self.N, self.d).transpose(1, 2)
        preV = self.Wv(x).view(B, S, self.N, self.d).transpose(1, 2)
        if kv_cache is not None and kv_cache.K is not None and kv_cache.V is not None:
            K = torch.cat([kv_cache.K, preK], dim=-2)
            V = torch.cat([kv_cache.V, preV], dim=-2)
        else:
            K, V = preK, preV
        if kv_cache is not None:
            kv_cache.append(preK, preV)

        qkt = torch.einsum("bnsd,bntd->bnst", Q, K) / (self.d ** 0.5)
        mask = torch.triu(
            qkt.new_full((Q.shape[-2], K.shape[-2]), -float("inf")),
            diagonal=(1 + K.shape[-2] - Q.shape[-2])
        )
        qkt += mask
        alpha = F.softmax(qkt, dim=-1)
        att = torch.einsum("bnst,bntd->bsnd", alpha, V).contiguous().view(B, S, self.H)
        out = self.Wo(att)
        assert out.shape == (B, S, self.D), [out.shape, (B, S, self.D)]
        return out
    
if __name__ == "__main__":
    # TODO: please improve tests.
    torch.manual_seed(42)
    B = 16
    D = 256
    H = 512
    N = 4
    S = 1024
    x = torch.randn(size=(B, S, D))
    net = MHA(D, H, N)
    out = net(x, KvCache())
    print(out.shape)
    print(out)