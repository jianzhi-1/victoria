import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    """
    As a reminder:
    - S ∈ R^{dv, dk} = Sum vi ki^T; represents memory
    - The hope is that ki, kj are near-orthogonal for i != j.
    - Sq represents value retrieved by querying q.
    """

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

    def forward(self, x: torch.Tensor, prev_S: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        In Maths:
        - S_t = alpha_t * (S_{t - 1} - beta_t * S_{t - 1} k_t k_t^T) + beta_t * v_t k_t^T
        - Bracketed part of first term represents the subtraction part of the delta rule.
        - Second term represents the addition part of the delta rule.
        - Alpha_t represents the retention gate.
        - Beta_t represents the update gate.
        """
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]
        assert prev_S.shape == (B, self.DV, self.DK), [prev_S.shape, (B, self.DV, self.DK)]
        if mask is None:
            mask = x.new_ones(size=(B, 1))
        assert mask.shape == (B, 1), [mask.shape, (B, 1)]
        mask = mask.view(B, 1, 1)
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        alpha = torch.sigmoid(self.Wa(x)).view(B, 1, 1)
        beta = torch.sigmoid(self.Wb(x)).view(B, 1, 1)
        S = prev_S - (1. - alpha) * mask * prev_S - alpha * beta * mask * torch.einsum("bv,bk->bvk", torch.einsum("bvk,bk->bv", prev_S, K), K) + beta * mask * torch.einsum("bv,bk->bvk", V, K)
        out = torch.einsum("bvk,bk->bv", S * mask, Q)
        return out, S

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 16
    DV = 32
    D = 8
    DK = 16
    gdn = GDN(D, DV, DK)
    S = torch.zeros(B, DV, DK, requires_grad=False)
    MIN_SEQ_LEN = 5
    MAX_SEQ_LEN = 512
    L = MAX_SEQ_LEN
    SEQ_LEN = F.one_hot(torch.randint(MIN_SEQ_LEN, MAX_SEQ_LEN, (B, 1)), num_classes=MAX_SEQ_LEN)
    mask = torch.arange(MAX_SEQ_LEN)[None, :] < torch.randint(MIN_SEQ_LEN, MAX_SEQ_LEN, (B, 1))
    for i in range(L):
        x = torch.randn(size=(B, D))
        out, S = gdn(x, S, mask[:,i:(i+1)])
        print(out)
    print("DONE")