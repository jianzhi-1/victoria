import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime

##### Basic imports
from streaming_dataset import StreamingDataset, get_splits
from basic.basic_net import BasicNet
from base.accumulator import Accumulator

@dataclass(frozen=True)
class ExperimentSetup:
    num_epochs: int
    B: int
    D: int
    H: int
    epochs_per_eval: int
    epochs_per_checkpoint: int
    seed: int
    lr: float
    max_gradient_norm: float = float("inf")

def checkpointing(
    PATH: str | Path,
    epoch: int,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    experiment_setup: ExperimentSetup
) -> None:
    to_serialize = {
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "experiment_setup": asdict(experiment_setup)
    }
    torch.save(to_serialize, PATH)

def train(setup: ExperimentSetup):
    B, D, H, seed, NUM_EPOCHS, epochs_per_eval, epochs_per_checkpoint, max_gradient_norm, lr = setup.B, setup.D, setup.H, setup.seed, setup.num_epochs, setup.epochs_per_eval, setup.epochs_per_checkpoint, setup.max_gradient_norm, setup.lr
    torch.manual_seed(seed)
    dataset = StreamingDataset(path_prefix="./data/data")
    ratios = {
        "train": 0.7,
        "eval": 0.3
    }
    train_dataset, eval_dataset = get_splits(dataset, ratios, seed)
    train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=B), DataLoader(eval_dataset, batch_size=B)

    print("train")
    for X, y in train_dataloader:
        print(X.shape, y.shape)

    print("eval")
    for X, y in eval_dataloader:
        print(X.shape, y.shape)

    net = BasicNet(D, H, 1)
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    grad_norm_acc = Accumulator()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        grad_norm_acc.clear()
        net.train()
        total_loss = 0.
        total_n = 0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            yhat = net(X)
            loss = loss_fn(yhat, y)
            total_loss += loss.item()
            total_n += X.shape[0]
            loss.backward()
            raw_gradient_norm = nn.utils.clip_grad_norm_(
                net.parameters(),
                max_norm=max_gradient_norm * X.shape[0]
            )
            gradient_norm: float = raw_gradient_norm.item() / X.shape[0]
            grad_norm_acc.push(gradient_norm)
            optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()
        print(f"[train] {epoch}/{NUM_EPOCHS}: avg loss {total_loss / total_n}; grad_norm_mean {grad_norm_acc.mean()}; grad_norm_p95 {grad_norm_acc.quantile(0.95)}; lr {current_lr}")

        if epoch % epochs_per_eval == 0 or epoch == NUM_EPOCHS:
            # Eval
            net.eval()
            with torch.no_grad():
                total_loss = 0.
                total_n = 0
                for X, y in eval_dataloader:
                    yhat = net(X)
                    loss = loss_fn(yhat, y)
                    total_loss += loss.item()
                    total_n += X.shape[0]
                print(f"[eval] {epoch}/{NUM_EPOCHS}: avg loss {total_loss / total_n}")
        
        if epoch % epochs_per_checkpoint == 0 or epoch == NUM_EPOCHS:
            checkpointing(
                f"./models/checkpoint_{epoch}_{NUM_EPOCHS}_basic.pth",
                epoch,
                net,
                optimizer,
                lr_scheduler,
                setup
            )

if __name__ == "__main__":
    setup = ExperimentSetup(
        100,
        16,
        16,
        256,
        10,
        50,
        42,
        1e-3
    )
    train(setup)



