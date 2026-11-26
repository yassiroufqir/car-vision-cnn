import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tr = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), norm])
    va = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    train = datasets.ImageFolder(f"{a.data}/train", tr)
    val = datasets.ImageFolder(f"{a.data}/val", va)
    tl = DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=2)
    vl = DataLoader(val, batch_size=a.bs, num_workers=2)

    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(train.classes))
    m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(a.epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(m(x), y).backward()
            opt.step()
        m.eval()
        ok = n = 0
        with torch.no_grad():
            for x, y in vl:
                p = m(x.to(dev)).argmax(1).cpu()
                ok += (p == y).sum().item()
                n += len(y)
        acc = ok / n
        print(f"epoch {ep+1}/{a.epochs} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": m.state_dict(), "classes": train.classes}, "models/classifier.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/classification")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    main(p.parse_args())
