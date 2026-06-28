import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

##### Basic imports
from streaming_dataset import StreamingDataset, get_splits
from basic.basic_net import BasicNet

def train(B: int):
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
    NUM_EPOCHS = 10
    net = BasicNet(D, 1)
    net.train()
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(NUM_EPOCHS):
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
        print(f"epoch {epoch} done; avg loss {total_loss / total_n}")

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
        print(f"validation done; avg loss {total_loss / total_n}")
    
    torch.save(net.state_dict(), "./models/basic.pth")

if __name__ == "__main__":
    train(16)



