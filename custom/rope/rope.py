import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, D: int, theta: float = 10000, MAX_SEQ_LEN: int = 4096) -> None:
        super().__init__()
        assert D % 2 == 0, D
        self.D = D
        self.theta = theta
        FREQ_TABLE = torch.outer(
            torch.arange(MAX_SEQ_LEN).float(),
            1. / (theta ** (torch.arange(0, D, 2).float() / D))
        )
        assert FREQ_TABLE.shape == (MAX_SEQ_LEN, D // 2)
        self.register_buffer("cos", FREQ_TABLE.cos())
        self.register_buffer("sin", FREQ_TABLE.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: support (B, H, S, D)
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]
        cos = self.cos[:S].view(1, S, self.D//2)
        sin = self.sin[:S].view(1, S, self.D//2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.stack(
            (x1 * cos - x2 * sin, x1 * sin + x2 * cos),
            dim=-1
        ).flatten(-2)
        assert out.shape == (B, S, self.D), [out.shape, (B, S, self.D)]
        return out

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 32
    S = 256
    D = 128
    rope = RoPE(D)
    x = torch.randn(size=(B, S, D))
    out = rope(x)

    import numpy as np
    D = 4
    x = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, -2.0, 3.0, -4.0]
    ])
    def theta(s: int, d: int, D: int) -> float:
        assert d % 2 == 0, d
        THETA = 10000
        return s / (THETA ** (d / D))

    def rotation_matrix(s: int, d: int, D: int) -> np.ndarray:
        t = theta(s, d, D)
        return np.array([
            [np.cos(t), -np.sin(t)],
            [np.sin(t), np.cos(t)]
        ])

    rope_x = np.array([
        np.concatenate([rotation_matrix(s, d, D) @ np.array([v[d], v[d + 1]]) for d in range(0, len(v), 2) ], axis=0)
        for s, v in enumerate(x)
    ])
    expected = torch.as_tensor(rope_x)
    actual = RoPE(4)(torch.as_tensor(x).unsqueeze(0))
    assert torch.allclose(expected, actual), (expected - actual).abs().max().item()
