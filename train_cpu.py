import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

##### Basic imports
from streaming_dataset import StreamingDataset, get_splits
from basic.basic_net import BasicNet


def train(B: int, epochs_per_checkpoint: int, epochs_per_eval: int):
    torch.manual_seed(42)
    dataset = StreamingDataset(path_prefix="./data/data")
    ratios = {
        "train": 0.7,
        "eval": 0.3
    }
    seed = 42
    train_dataset, eval_dataset = get_splits(dataset, ratios, seed)
    train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=B), DataLoader(eval_dataset, batch_size=B)

    print("train")
    for X, y in train_dataloader:
        print(X.shape, y.shape)

    print("eval")
    for X, y in eval_dataloader:
        print(X.shape, y.shape)

    D = 16
    NUM_EPOCHS = 100
    net = BasicNet(D, 1)
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(NUM_EPOCHS):
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
            optimizer.step()
        print(f"[train] {epoch}/{NUM_EPOCHS}: avg loss {total_loss / total_n}")

        if (epoch + 1) % epochs_per_eval == 0 or (epoch + 1) == NUM_EPOCHS:
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
        
        if (epoch + 1) % epochs_per_checkpoint == 0:
            torch.save(net.state_dict(), f"./models/checkpoint_{epoch}_{NUM_EPOCHS}_basic.pth")

    torch.save(net.state_dict(), f"./models/checkpoint_{NUM_EPOCHS}_{NUM_EPOCHS}_basic.pth")

if __name__ == "__main__":
    train(16, 50, 10)



