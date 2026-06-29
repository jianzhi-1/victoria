import torch
import torch.distributed as dist
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class DDPInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    def unpack(self) -> tuple[int, int, int, torch.device]:
        return self.rank, self.world_size, self.local_rank, self.device

def ddp_setup() -> DDPInfo:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return DDPInfo(
        dist.get_rank(),
        dist.get_world_size(),
        local_rank=local_rank,
        device=torch.device(f"cuda:{local_rank}")
    )

def d_reduce(v: float, device: torch.device, op: dist.ReduceOp.RedOpType) -> float:
    """
    Each device has its own storage for v.
    At the end, t = reduce(v), consistent across all devices.
    """
    t = torch.tensor(v, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=op)
    return t.item()